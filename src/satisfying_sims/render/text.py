from dataclasses import dataclass, field

@dataclass
class TextConfig:
    font_size: int = 14
    font_color: str = "white"
    font_family: str = "sans-serif"
    world_text_pad: float =0.015
    line_gap: float = 0.07
    layer: int = 0
    
    use_hud_text: bool = True
    hud_size: int = 14
    
    use_caption_text: bool = True
    caption_size: int = 16
    caption_content: str = ""
    
    use_debug_text: bool = False
    debug_size: int = 14