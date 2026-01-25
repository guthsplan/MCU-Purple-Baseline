# src/agent/download.py
"""
Pre-download VPT models from Hugging Face to avoid runtime delays.
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger("purple.download")


def download_vpt_models(
    hf_id_rl: str = "CraftJarvis/MineStudio_VPT.rl_from_early_game_2x",
    hf_id_fallback: str = "CraftJarvis/MineStudio_VPT.foundation_2x",
    device: Optional[str] = None,
) -> None:
    """
    Pre-download VPT models from Hugging Face.
    
    This function instantiates VPTPolicy to trigger model download,
    then discards the instance. The actual model will be loaded later
    in VPTAgent.__init__().
    
    Parameters
    ----------
    hf_id_rl : str
        Hugging Face model ID for RL model
    hf_id_fallback : str
        Hugging Face model ID for fallback foundation model
    device : Optional[str]
        Device spec (not critical for download, but passed for consistency)
    """
    import torch
    from minestudio.models.vpt import VPTPolicy
    from src.action.action_space import build_vpt_action_space
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    action_space = build_vpt_action_space()
    
    logger.info("Pre-downloading VPT RL model: %s", hf_id_rl)
    try:
        model_rl = VPTPolicy.from_pretrained(
            hf_id_rl,
            action_space=action_space,
        )
        logger.info("VPT RL model downloaded successfully")
        del model_rl  # Free memory
    except Exception as e:
        logger.warning("Failed to download VPT RL model (%s): %s", hf_id_rl, e)
        logger.info("Downloading VPT fallback model: %s", hf_id_fallback)
        try:
            model_fallback = VPTPolicy.from_pretrained(
                hf_id_fallback,
                action_space=action_space,
            )
            logger.info("VPT fallback model downloaded successfully")
            del model_fallback  # Free memory
        except Exception as e2:
            logger.error("Failed to download VPT fallback model (%s): %s", hf_id_fallback, e2)
            raise


def download_steve1_models(device: Optional[str] = None) -> None:
    """
    Pre-download STEVE-1 models from Hugging Face.
    """
    import torch
    from minestudio.models.steve_one.body import SteveOnePolicy
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info("Pre-downloading STEVE-1 model: CraftJarvis/MineStudio_STEVE-1.12w_EMA")
    try:
        model = SteveOnePolicy.from_pretrained(
            "CraftJarvis/MineStudio_STEVE-1.12w_EMA"
        )
        logger.info("STEVE-1 model downloaded successfully")
        del model
    except Exception as e:
        logger.error("Failed to download STEVE-1 model: %s", e)
        raise


def download_rocket1_models(device: Optional[str] = None) -> None:
    """
    Pre-download Rocket-1 models from Hugging Face.
    """
    import torch
    from minestudio.models.rocket_one.body import RocketPolicy
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info("Pre-downloading Rocket-1 model: CraftJarvis/MineStudio_ROCKET-1.12w_EMA")
    try:
        model = RocketPolicy.from_pretrained(
            "CraftJarvis/MineStudio_ROCKET-1.12w_EMA"
        )
        logger.info("Rocket-1 model downloaded successfully")
        del model
    except Exception as e:
        logger.error("Failed to download Rocket-1 model: %s", e)
        raise


def download_groot1_models(device: Optional[str] = None) -> None:
    """
    Pre-download Groot-1 models from Hugging Face.
    """
    import torch
    from minestudio.models.groot_one.body import GrootPolicy
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info("Pre-downloading Groot-1 model: CraftJarvis/MineStudio_GROOT.18w_EMA")
    try:
        model = GrootPolicy.from_pretrained(
            "CraftJarvis/MineStudio_GROOT.18w_EMA"
        )
        logger.info("Groot-1 model downloaded successfully")
        del model
    except Exception as e:
        logger.error("Failed to download Groot-1 model: %s", e)
        raise


def download_models_for_agent(agent_name: str, device: Optional[str] = None) -> None:
    """
    Download models based on agent type.
    
    Parameters
    ----------
    agent_name : str
        Agent identifier ("vpt", "steve1", "rocket1", etc.)
    device : Optional[str]
        Device spec for model loading
    """
    name = (agent_name or "noop").lower()
    
    if name == "vpt":
        download_vpt_models(device=device)
    elif name == "steve1":
        download_steve1_models(device=device)
    elif name == "rocket1":
        download_rocket1_models(device=device)
    elif name == "groot1":
        download_groot1_models(device=device)
    elif name == "noop":
        logger.info("NoOp agent doesn't require model downloads")
    elif name == "llm":
        logger.info("LLM agent doesn't require model downloads (uses API)")
    else:
        logger.warning("Unknown agent type: %s - skipping model download", name)

if __name__ == "__main__":
    """CLI entry point for Docker build-time model downloads."""
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    
    parser = argparse.ArgumentParser(
        description="Pre-download models from Hugging Face for MCU Purple Agent"
    )
    parser.add_argument(
        "--agent",
        default="vpt",
        choices=["vpt", "steve1", "rocket1", "groot1", "noop", "llm"],
        help="Agent type to download models for",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for model initialization (cpu/cuda)",
    )
    args = parser.parse_args()
    
    logger.info("Starting model download for agent: %s", args.agent)
    try:
        download_models_for_agent(args.agent, device=args.device)
        logger.info("Model download completed successfully")
    except Exception as e:
        logger.error("Model download failed: %s", e)
        raise
