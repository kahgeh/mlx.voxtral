#!/usr/bin/env python3
"""
MLX-Voxtral Chat Once CLI

A command-line interface for single-turn chat with audio using MLX-Voxtral models.

Usage:
    uv run mlx-voxtral.chat_once "What is in this audio?" path/to/audio.wav
"""

import argparse
import sys
import mlx.core as mx
from mlx_voxtral import VoxtralProcessor, load_voxtral_model


def main():
    parser = argparse.ArgumentParser(
        description="Single-turn chat with audio using MLX-Voxtral models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simple transcription
  uv run mlx-voxtral.chat_once "Transcribe this audio" audio.mp3

  # Ask about audio content
  uv run mlx-voxtral.chat_once "What language is being spoken?" speech.wav

  # Transcribe with context
  uv run mlx-voxtral.chat_once "Please transcribe this nursery rhyme:" nursery_rhyme.mp3

  # Use audio from URL
  uv run mlx-voxtral.chat_once "What is being said?" https://example.com/audio.mp3
        """
    )
    
    parser.add_argument(
        "prompt",
        type=str,
        help="The prompt/question about the audio"
    )
    
    parser.add_argument(
        "audio",
        type=str,
        help="Path to audio file or URL"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="mzbac/voxtral-mini-3b-4bit-mixed",
        help="Model name or path (default: mzbac/voxtral-mini-3b-4bit-mixed)"
    )
    
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=150,
        help="Maximum number of tokens to generate (default: 150)"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature, 0.0 for deterministic output (default: 0.0)"
    )
    
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Nucleus sampling probability threshold (default: 0.95)"
    )
    
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Top-k sampling, 0 to disable (default: 0)"
    )
    
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Model dtype (default: bfloat16)"
    )
    
    args = parser.parse_args()
    
    dtype_map = {
        "float32": mx.float32,
        "float16": mx.float16,
        "bfloat16": mx.bfloat16
    }
    dtype = dtype_map[args.dtype]
    
    # Load model and processor
    print(f"Loading model: {args.model}", file=sys.stderr)
    model, config = load_voxtral_model(args.model, dtype=dtype)
    processor = VoxtralProcessor.from_pretrained(args.model)
    
    # Create conversation
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": args.prompt},
                {"type": "audio", "url": args.audio}
            ]
        }
    ]
    
    # Apply chat template
    inputs = processor.apply_chat_template(conversation, return_tensors="mlx")
    
    # Generate response
    outputs = model.generate(
        inputs["input_ids"],
        input_features=inputs["input_features"],
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        do_sample=args.temperature > 0,
        top_p=args.top_p,
        top_k=args.top_k,
    )
    
    # Decode and print only the response
    response = processor.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(response)


if __name__ == "__main__":
    main()