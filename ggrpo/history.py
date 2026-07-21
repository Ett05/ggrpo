from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class GRPOHistory:
    """
    Holds training metrics logged during GRPO fine-tuning sessions.
    """
    epoch_rewards: List[float] = field(default_factory=list)
    epoch_losses: List[float] = field(default_factory=list)
    kl_divergences: List[float] = field(default_factory=list)

    def plot(self, save_path: Optional[str] = None):
        """
        Plots loss and reward curves over epochs.
        """
        try:
            import matplotlib.pyplot as plt
            
            fig, ax1 = plt.subplots(figsize=(8, 5))

            color = 'tab:blue'
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Reward', color=color)
            ax1.plot(self.epoch_rewards, color=color, label='Avg Reward', marker='o')
            ax1.tick_params(axis='y', labelcolor=color)

            if self.epoch_losses:
                ax2 = ax1.twinx()  
                color = 'tab:red'
                ax2.set_ylabel('Loss', color=color)
                ax2.plot(self.epoch_losses, color=color, label='Loss', linestyle='--')
                ax2.tick_params(axis='y', labelcolor=color)

            plt.title("ggrpo Fine-tuning Metrics")
            fig.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
                print(f"Plot saved to {save_path}")
            else:
                plt.show()
        except ImportError:
            print("matplotlib is required for history.plot(). Install it with `pip install matplotlib`.")
