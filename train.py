import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from einops import rearrange
import math
from tqdm.auto import tqdm
from pathlib import Path

from encoder import VitEncoder, VitEncoderConfig
from decoder import VitDecoder, VitDecoderConfig
from src.data.data import VideoDataset
from slotssm import SlotSSM, SlotSSMConfig

logger = get_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Train SlotSSM with VIT encoder/decoder")
    
    # Model architecture
    parser.add_argument("--num_slots", type=int, default=6)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--patch_size", type=int, default=8)
    parser.add_argument("--num_channels", type=int, default=3)
    parser.add_argument("--num_encoder_layers", type=int, default=6)
    parser.add_argument("--num_decoder_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_ssm_blocks", type=int, default=4)
    
    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    
    # Data and logging
    parser.add_argument("--train_data_path", type=str, required=True, default="/path-to-your-dataset")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument("--save_steps", type=int, default=10)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    
    return parser.parse_args()

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def main():
    args = parse_args()
    
    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_dir=args.output_dir,
    )

    # Set random seed for reproducibility
    set_seed(args.seed)

    # Create output directory
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # Log basic info
    logger.info("Starting training with config:")
    for key, value in vars(args).items():
        logger.info(f"{key}: {value}")

    # Define model configurations and initialize models
    encoder_config = VitEncoderConfig(
        image_size=args.image_size,
        patch_size=args.patch_size,
        num_channels=args.num_channels,
        hidden_size=args.d_model,
        num_hidden_layers=args.num_encoder_layers,
        num_attention_heads=args.num_heads,
        output_dim=args.d_model
    )

    decoder_config = VitDecoderConfig(
        input_size=args.d_model,
        hidden_size=args.d_model,
        num_hidden_layers=args.num_decoder_layers,
        num_attention_heads=args.num_heads,
        image_size=args.image_size,
        patch_size=args.patch_size,
        num_channels=args.num_channels,
        num_cls_tokens=args.num_slots
    )

    encoder = VitEncoder(encoder_config)
    decoder = VitDecoder(decoder_config)
    slotssm_config = SlotSSMConfig(
        num_slots=args.num_slots,
        num_blocks=args.num_ssm_blocks,
        d_model=args.d_model,
        use_cross_attn=True,
        space_attn_num_heads=args.d_model // 64,
        input_d_model=args.d_model,
        use_inverted_attention=True,
        encoder_attn_num_heads=args.d_model // 64
    )
    slotssm = SlotSSM(slotssm_config)

    # Setup optimizer
    params = list(encoder.parameters()) + list(decoder.parameters()) + list(slotssm.parameters())
    optimizer = Adam(params, lr=args.learning_rate)

    # Setup dataset and dataloader
    train_dataset = VideoDataset(
        root=args.train_data_path,
        seq_len=args.seq_len, 
        resolution=args.image_size,
        image_norm_mode="imagenet"
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
    )

    # Calculate number of update steps per epoch
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
        overrode_max_train_steps = True

    # Setup learning rate scheduler
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_train_steps
    )

    # Prepare everything with accelerator
    encoder, decoder, slotssm, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        encoder, decoder, slotssm, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(
        args.max_train_steps / num_update_steps_per_epoch)

    # Define save/load hooks for model checkpointing
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for model in models:
                # Check if model is one of our models
                if not isinstance(model, (type(accelerator.unwrap_model(encoder)), 
                                       type(accelerator.unwrap_model(decoder)), 
                                       type(accelerator.unwrap_model(slotssm)))):
                    continue
                
                sub_dir = model._get_name().lower()
                model.save_pretrained(os.path.join(output_dir, sub_dir))
                
                # Pop weight so corresponding model is not saved again
                weights.pop()

    def load_model_hook(models, input_dir):
        while len(models) > 0:
            # Pop models so they are not loaded again
            model = models.pop()
            sub_dir = model._get_name().lower()

            if isinstance(model, (type(accelerator.unwrap_model(encoder)), 
                                  type(accelerator.unwrap_model(decoder)), 
                                  type(accelerator.unwrap_model(slotssm)))):
                load_model = type(model).from_pretrained(
                    input_dir, subfolder=sub_dir)
                # model.register_to_config(**load_model.config)
                model.config = load_model.config
            else:
                raise ValueError(f"Unknown model type {type(model)}")

            model.load_state_dict(load_model.state_dict())
            del load_model

    # Register the hooks with accelerator
    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Initialize training variables
    global_step = 0
    first_epoch = 0
    accumulate_steps = 0
    # Resume from checkpoint if specified
    if args.resume_from_checkpoint is not None and os.path.exists(args.resume_from_checkpoint):

        checkpoint_path = args.resume_from_checkpoint
        path = os.path.basename(args.resume_from_checkpoint)

        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.load_state(checkpoint_path)
        
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            first_epoch = int(training_difference.replace("epoch_", "")) + 1
            global_step = first_epoch * num_update_steps_per_epoch
        else:
            global_step = int(training_difference.replace("step_", ""))
            first_epoch = global_step // num_update_steps_per_epoch

        initial_global_step = global_step
        accumulate_steps = global_step * args.gradient_accumulation_steps
    else:
        initial_global_step = 0

    # Setup progress bar
    progress_bar = tqdm(
        range(args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
        position=0,
        leave=True,
        ncols=100
    )

    accelerator.wait_for_everyone()

    # Training loop
    for epoch in range(first_epoch, args.num_train_epochs):
        encoder.train()
        decoder.train()
        slotssm.train()
        
        for step, batch in enumerate(train_dataloader):
            frames = batch["pixel_values"]
            B, T, C, H, W = frames.shape
            
            with accelerator.accumulate(encoder, decoder, slotssm):
                # Forward pass
                frames_flat = rearrange(frames, 'b t c h w -> (b t) c h w')
                encoder_out = encoder(frames_flat)
                features = encoder_out.last_hidden_state
                features = rearrange(features, '(b t) n d -> b t n d', b=B, t=T)
                
                slotssm_out = slotssm(features)
                slots = slotssm_out.slots
                
                decoder_outputs = decoder(rearrange(slots, 'b t n d -> (b t) n d'))
                recon = decoder_outputs.recon
                recon = rearrange(recon, '(b t) c h w -> b t c h w', b=B, t=T)
                
                loss = nn.MSELoss()(recon, frames)
                
                # Backward pass
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params, args.grad_clip)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
                if accelerator.sync_gradients:     
                    # Only update global_step when we actually apply gradients
                    global_step += 1
                    
                    # Update progress bar and save checkpoint if needed
                    progress_bar.set_postfix(
                        loss=f"{loss.item():.4f}",
                        lr=f"{lr_scheduler.get_last_lr()[0]:.6f}",
                        epoch=f"{epoch+1}/{args.num_train_epochs}"
                    )
                    progress_bar.update(1)
                    
                    if global_step % args.save_steps == 0:
                        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint_step_{global_step}")
                        accelerator.save_state(checkpoint_dir)
                        accelerator.print(f"Successfully saved checkpoint at step {global_step}")
            
            accumulate_steps += 1
            
            if global_step >= args.max_train_steps:
                break

    # Save final checkpoint
    final_checkpoint_dir = os.path.join(args.output_dir, "final_checkpoint")
    accelerator.save_state(final_checkpoint_dir)
    # logger.info(f"Successfully saved final checkpoint to {final_checkpoint_dir}")
    accelerator.print(f"Successfully saved final checkpoint to {final_checkpoint_dir}")

if __name__ == "__main__":
    main()

# CUDA_VISIBLE_DEVICES=2 accelerate launch --num_processes=1 --main_process_port 29510 release/train.py --train_data_path /data/local/jindong/Datasets/clevrer \
# --resume_from_checkpoint /common/users/jj691/Documents/Programs/PyTorch/slot-ssm/outputs/checkpoint-10