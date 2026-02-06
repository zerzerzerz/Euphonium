"""Video generation configuration."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class VideoArguments:
    """Video generation related configuration."""
    
    h: Optional[int] = field(
        default=None,
        metadata={"help": "Video height"}
    )
    
    w: Optional[int] = field(
        default=None,
        metadata={"help": "Video width"}
    )
    
    t: Optional[int] = field(
        default=None,
        metadata={"help": "Video length (number of frames)"}
    )
    
    fps: Optional[int] = field(
        default=None,
        metadata={"help": "Frame rate of the output video"}
    )
