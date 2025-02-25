import lionheart
import torch
from lionheart.trainer_evaluator.MobileBERTSquad import MobileBERTSquad
from lionheart.methods.LH import LH, LHConfig

if __name__ == "__main__":
    config = LHConfig(
        trainer_evaluator=MobileBERTSquad(),
        checkpoint_path='cp.pt',
        train_batch_size=4,
        eval_batch_size=4,
        digital_lr=1e-2,
        digital_momentum=0.85,
        analog_lr=1e-3,
        analog_momentum=0.85,
        drop_threshold=2.,
        t_eval=86400.0,
        evaluation_reps=2,
        patience=2,
        num_steps=50,
    )
    lh = LH(config)
    lh.set_baseline_score()
    lh.run()