from .backbones.swin_prompt import PromptSwinTransformer
from mmseg.models.builder import BACKBONES

BACKBONES.register_module()(PromptSwinTransformer)

__all__ = ['PromptSwinTransformer']