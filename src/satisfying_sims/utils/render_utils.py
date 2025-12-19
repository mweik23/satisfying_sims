

def fig_inches_from_pixels(width_px: int | None = None, 
                           height_px: int | None = None, 
                           dpi: int = 100, 
                           figsize_default: tuple[float, float] = (6.0, 6.0)) -> tuple[float, float]:
    if width_px is not None and height_px is not None:
        return (width_px / dpi, height_px / dpi)
    else:
        return figsize_default