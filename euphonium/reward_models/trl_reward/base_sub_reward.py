"""
Base Class for Sub-reward Models

All sub-reward models integrated into TRLRewardModel should inherit from this base class.
"""

import torch
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any


class BaseSubReward(ABC):
    """
    Abstract base class for sub-reward models.
    
    Defines common interfaces and configuration parsing logic for sub-reward models.
    """
    
    def __init__(self, config: Dict[str, Any], device: str = "cuda"):
        """
        Initialize the sub-reward model.
        
        Args:
            config: Model configuration parameter dictionary.
            device: Computing device.
        """
        self.config = config
        self.device = device
        self._enabled = False
        self._coef = 1.0
        
    @property
    def enabled(self) -> bool:
        """Whether this sub-reward is enabled."""
        return self._enabled
    
    @property
    def coef(self) -> float:
        """Reward coefficient."""
        return self._coef
        
    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize the model (if enabled).
        
        Subclasses should load model weights, etc., here.
        """
        pass
    
    @abstractmethod
    def compute(self, *args, **kwargs) -> Any:
        """
        Compute reward scores.
        
        Subclasses should override this method and define specific parameter signatures.
        
        Returns:
            Reward score (the specific type is defined by the subclass).
        """
        pass
    
    def _get_config_value(self, key: str, default: Any, value_type: type, 
                          print_default: bool = True) -> Any:
        """
        Get a value from configuration, or use the default if it doesn't exist.
        
        Args:
            key: Configuration key name.
            default: Default value.
            value_type: Value type (for conversion).
            print_default: Whether to print a message when using the default value.
            
        Returns:
            Configuration value or default value.
        """
        value = self.config.get(key, None)
        if value is None:
            if print_default:
                # Only print if default is meaningful (not empty string or None)
                if default not in (None, "", [], {}):
                    print(f"[{self.__class__.__name__}] {key} not specified, using default {default}")
            return default
        return value_type(value)
