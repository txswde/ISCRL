"""Train and evaluate ISCRL as specified in the accompanying manuscript."""

from __future__ import annotations

import argparse
import datetime
import os
from pathlib import Path
import random
import sys
import time
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import h5py
import numpy as np
from tabulate import tabulate
import torch
from torch import Tensor
from torch.distributions import Bernoulli

from iscrl_components import AIMController, SimCLRObjective
from models import ISCRLPolicy
from rewards import compute_reward
from scores.eval import evaluate_scores, generate_scores
from utils import Logger, mkdir_if_missing, read_json, save_checkpoint, write_json
import vsum_tools


DEFAULT_DATASETS = (
    "datasets/eccv16_dataset_summe_google_pool5.h5",
    "datasets/eccv16_dataset_tvsum_google_pool5.h5",
    "datasets/eccv16_dataset_ovp_google_pool5.h5",
    "datasets/eccv16_dataset_youtube_google_pool5.h5",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ISCRL: invariant semantic contrastive reinforcement learning"
    )
    data = parser.add_argument_group("data")
    data.add_argument(
        "-d",
        "--dataset",
        action="append",
        help=(
            "HDF5 feature file. Repeat for augmented/transfer experiments. "
            "When omitted, the four conventional paths under datasets/ are used."
        ),
    )
    data.add_argument("-s", "--split", required=True, help="JSON split file")
    data.add_argument("--split-id", type=int, default=0)
    data.add_argument("-us", "--userscore", help="optional HDF5 user-score file")
    data.add_argument("-m", "--metric", required=True, choices=("tvsum", "summe"))

    model = parser.add_argument_group("model")
    model.add_argument("--input-dim", type=int, default=1024)
    model.add_argument("--hidden-dim", type=int, default=512)
    model.add_argument("--invariant-dim", type=int, default=128)
    model.add_argument("--projector-hidden-dim", type=int, default=512)
    model.add_argument("--attention-dropout", type=float, default=0.5)

    contrastive = parser.add_argument_group("contrastive warm-up")
    contrastive.add_argument("--warmup-epochs", type=int, default=50)
    contrastive.add_argument("--temperature", type=float, default=0.5)
    contrastive.add_argument("--feature-noise", type=float, default=0.03)
    contrastive.add_argument("--feature-mask", type=float, default=0.20)
    contrastive.add_argument("--temporal-jitter", type=int, default=1)
    contrastive.add_argument("--warmup-lr", type=float, default=1e-5)

    rl = parser.add_argument_group("reinforcement learning")
    rl.add_argument("--rl-epochs", type=int, default=300)
    rl.add_argument("--lr", type=float, default=1e-5)
    rl.add_argument("--weight-decay", type=float, default=1e-5)
    rl.add_argument("--lr-step", type=int, default=30)
    rl.add_argument("--lr-gamma", type=float, default=0.5)
    rl.add_argument("--episodes", type=int, default=5)
    rl.add_argument(
        "--reward-beta",
        type=float,
        default=0.10,
        help="invariant-space contribution to the dual reward",
    )
    rl.add_argument("--temporal-threshold", type=int, default=20)
    rl.add_argument("--baseline-decay", type=float, default=0.90)

    aim = parser.add_argument_group("adaptive intervention mechanism")
    aim.add_argument("--intervention-min", type=float, default=0.10)
    aim.add_argument("--intervention-max", type=float, default=0.50)
    aim.add_argument("--intervention-penalty", type=float, default=0.10)
    aim.add_argument("--intervention-increase", type=float, default=1.10)
    aim.add_argument("--intervention-decrease", type=float, default=0.90)
    aim.add_argument("--base-clip", type=float, default=5.0)

    misc = parser.add_argument_group("runtime")
    misc.add_argument("--seed", type=int, default=1)
    misc.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    misc.add_argument("--evaluate", action="store_true")
    misc.add_argument("--save-dir", default="log")
    misc.add_argument("--resume", default="")
    misc.add_argument("--verbose", action="store_true")
    misc.add_argument("--save-results", action="store_true")
    return parser


class DatasetCollection:
    """Resolve both canonical keys and ``dataset_name/video_N`` split keys."""

    def __init__(self, paths: Sequence[str]) -> None:
        if not paths:
            raise ValueError("at least one dataset is required")
        self.files: Dict[str, h5py.File] = {}
        for path in paths:
            name = Path(path).stem
            if name in self.files:
                raise ValueError(f"duplicate dataset stem: {name}")
            self.files[name] = h5py.File(path, "r")

    def resolve(self, split_key: str) -> h5py.Group:
        if "/" in split_key:
            dataset_name, video_key = split_key.split("/", 1)
            if dataset_name not in self.files:
                raise KeyError(f"split references unopened dataset {dataset_name!r}")
            return self.files[dataset_name][video_key]
        if len(self.files) == 1:
            return next(iter(self.files.values()))[split_key]
        matches = [dataset[split_key] for dataset in self.files.values() if split_key in dataset]
        if len(matches) != 1:
            raise KeyError(
                f"unqualified key {split_key!r} resolves to {len(matches)} datasets; "
                "use dataset_name/video_key in the split"
            )
        return matches[0]

    def close(self) -> None:
        for dataset in self.files.values():
            dataset.close()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_split(path: str, split_id: int) -> Tuple[List[str], List[str]]:
    splits = read_json(path)
    if not 0 <= split_id < len(splits):
        raise IndexError(f"split-id {split_id} is outside [0, {len(splits) - 1}]")
    split = splits[split_id]
    return split["train_keys"], split["test_keys"]


def feature_tensor(group: h5py.Group, device: torch.device) -> Tensor:
    features = np.asarray(group["features"], dtype=np.float32)
    return torch.from_numpy(features).unsqueeze(0).to(device)


def contrastive_warmup(
    model: ISCRLPolicy,
    datasets: DatasetCollection,
    train_keys: Sequence[str],
    args: argparse.Namespace,
    device: torch.device,
) -> List[float]:
    """Optimize only the 1024-512-128 projection head for 50 epochs."""
    if args.warmup_epochs <= 0:
        return []
    model.set_projector_trainable(True)
    objective = SimCLRObjective(
        model.simclr_projector,
        temperature=args.temperature,
        noise_std=args.feature_noise,
        mask_prob=args.feature_mask,
        temporal_jitter=args.temporal_jitter,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.simclr_projector.parameters(),
        lr=args.warmup_lr,
        weight_decay=args.weight_decay,
    )
    history: List[float] = []
    print(f"==> Contrastive warm-up ({args.warmup_epochs} epochs)")
    for epoch in range(args.warmup_epochs):
        losses: List[float] = []
        order = np.random.permutation(len(train_keys))
        for index in order:
            original = feature_tensor(datasets.resolve(train_keys[index]), device)
            if original.size(1) < 2:
                continue
            optimizer.zero_grad(set_to_none=True)
            loss = objective(original)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        mean_loss = float(np.mean(losses)) if losses else float("nan")
        history.append(mean_loss)
        print(
            f"Warm-up {epoch + 1:03d}/{args.warmup_epochs:03d} "
            f"InfoNCE {mean_loss:.6f}"
        )
    model.set_projector_trainable(False)
    return history


def scheduled_learning_rate(args: argparse.Namespace, epoch: int) -> float:
    if args.lr_step <= 0:
        return args.lr
    return args.lr * args.lr_gamma ** (epoch // args.lr_step)


def train_policy(
    model: ISCRLPolicy,
    datasets: DatasetCollection,
    train_keys: Sequence[str],
    args: argparse.Namespace,
    device: torch.device,
    aim: AIMController,
    start_epoch: int = 0,
    optimizer_state: Mapping[str, object] | None = None,
) -> Tuple[Mapping[str, List[float]], torch.optim.Optimizer]:
    """Run Algorithm 1 with clipping and LR scaling applied before each update."""
    model.set_projector_trainable(False)
    policy_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(
        policy_parameters,
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.5, 0.999),
    )
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    reward_history: Dict[str, List[float]] = {key: [] for key in train_keys}
    print(f"==> RL policy optimization ({args.rl_epochs} epochs)")

    for epoch in range(start_epoch, args.rl_epochs):
        model.train()
        epoch_rewards: List[float] = []
        epoch_losses: List[float] = []
        epoch_grad_norms: List[float] = []
        order = np.random.permutation(len(train_keys))
        base_lr = scheduled_learning_rate(args, epoch)

        for index in order:
            video_id = train_keys[index]
            original = feature_tensor(datasets.resolve(video_id), device)
            probabilities, _, _, invariant, _ = model(original)
            distribution = Bernoulli(probabilities.clamp(1e-6, 1.0 - 1e-6))
            baseline = aim.state_for(video_id).baseline
            losses: List[Tensor] = []
            rewards: List[float] = []

            for _ in range(args.episodes):
                actions = distribution.sample()
                reward = compute_reward(
                    original,
                    invariant,
                    actions,
                    beta=args.reward_beta,
                    temporal_distance_threshold=args.temporal_threshold,
                )
                reward_value = float(reward.detach().cpu())
                advantage = reward_value - baseline
                # Eq. (9) sums log-policy gradients over all sampled frames.
                losses.append(-distribution.log_prob(actions).sum() * advantage)
                rewards.append(reward_value)

            average_reward = float(np.mean(rewards))
            policy_loss = torch.stack(losses).mean()
            clip_threshold, lr_scale, _ = aim.update(video_id, average_reward)
            for group in optimizer.param_groups:
                group["lr"] = base_lr * lr_scale

            optimizer.zero_grad(set_to_none=True)
            policy_loss.backward()
            # AIM regulates the gradients used by this optimizer update.
            grad_norm = torch.nn.utils.clip_grad_norm_(policy_parameters, clip_threshold)
            optimizer.step()
            aim.update_baseline(video_id, average_reward)

            reward_history[video_id].append(average_reward)
            epoch_rewards.append(average_reward)
            epoch_losses.append(float(policy_loss.detach().cpu()))
            epoch_grad_norms.append(float(grad_norm.detach().cpu()))

        states = [aim.state_for(key) for key in train_keys]
        print(
            f"RL {epoch + 1:03d}/{args.rl_epochs:03d} "
            f"reward {np.mean(epoch_rewards):.6f} "
            f"loss {np.mean(epoch_losses):.6f} "
            f"grad {np.mean(epoch_grad_norms):.6f} "
            f"I {np.mean([state.intervention for state in states]):.4f} "
            f"base_lr {base_lr:.2e}"
        )
    return reward_history, optimizer


def _user_scores(userscores: h5py.File, split_key: str) -> np.ndarray:
    video_key = split_key.rsplit("/", 1)[-1]
    if split_key in userscores:
        group = userscores[split_key]
    elif video_key in userscores:
        group = userscores[video_key]
    else:
        raise KeyError(f"no user scores found for {split_key!r}")
    return np.asarray(group["user_scores"])


def evaluate(
    model: ISCRLPolicy,
    datasets: DatasetCollection,
    userscores: h5py.File | None,
    test_keys: Sequence[str],
    args: argparse.Namespace,
    device: torch.device,
) -> float:
    print("==> Evaluation")
    model.eval()
    eval_metric = "avg" if args.metric == "tvsum" else "max"
    f_scores: List[float] = []
    spearman: List[float] = []
    kendall: List[float] = []
    table = [["No.", "Video", "F-score"]]
    result_file = None
    if args.save_results:
        result_path = Path(args.save_dir) / (
            f"result_rl{args.rl_epochs}_split{args.split_id}.h5"
        )
        result_file = h5py.File(result_path, "w")

    with torch.no_grad():
        for number, split_key in enumerate(test_keys, start=1):
            group = datasets.resolve(split_key)
            original = feature_tensor(group, device)
            probabilities, _, _, _, _ = model(original)
            scores = probabilities.squeeze(0).squeeze(-1).cpu().numpy()
            change_points = np.asarray(group["change_points"])
            n_frames = int(group["n_frames"][()])
            n_frame_per_seg = np.asarray(group["n_frame_per_seg"]).tolist()
            picks = np.asarray(group["picks"])
            user_summary = np.asarray(group["user_summary"])
            gt_score = np.asarray(group["gtscore"])
            machine_summary, gt_frame_score = vsum_tools.generate_summary(
                scores,
                gt_score,
                change_points,
                n_frames,
                n_frame_per_seg,
                picks,
            )
            f_score, _, _ = vsum_tools.evaluate_summary(
                machine_summary, user_summary, eval_metric
            )
            f_scores.append(float(f_score))
            if userscores is not None:
                annotations = _user_scores(userscores, split_key)
                machine_scores = generate_scores(scores, n_frames, picks)
                spearman.append(evaluate_scores(machine_scores, annotations, "spearmanr"))
                kendall.append(evaluate_scores(machine_scores, annotations, "kendalltau"))
            if args.verbose:
                table.append([number, split_key, f"{f_score:.1%}"])
            if result_file is not None:
                output = result_file.require_group(split_key)
                output.create_dataset("gt_frame_score", data=gt_frame_score)
                output.create_dataset("score", data=scores)
                output.create_dataset("machine_summary", data=machine_summary)
                output.create_dataset("gtscore", data=gt_score)
                output.create_dataset("fm", data=f_score)

    if result_file is not None:
        result_file.close()
    if args.verbose:
        print(tabulate(table))
    mean_f_score = float(np.mean(f_scores))
    print(f"Average F1-score {mean_f_score:.1%}")
    if spearman:
        print(f"Average Kendall tau {np.nanmean(kendall):.6f}")
        print(f"Average Spearman rho {np.nanmean(spearman):.6f}")
    return mean_f_score


def checkpoint_state(
    model: ISCRLPolicy,
    optimizer: torch.optim.Optimizer | None,
    aim: AIMController,
    args: argparse.Namespace,
    warmup_history: Sequence[float],
) -> Dict[str, object]:
    return {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "aim": aim.state_dict(),
        "warmup_history": list(warmup_history),
        "rl_epoch": args.rl_epochs,
        "args": vars(args),
    }


def run(args: argparse.Namespace) -> None:
    if args.episodes < 1:
        raise ValueError("--episodes must be at least 1")
    seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    mkdir_if_missing(args.save_dir)
    log_name = "log_test.txt" if args.evaluate else "log_train.txt"
    logger = Logger(os.path.join(args.save_dir, log_name))
    original_stdout = sys.stdout
    sys.stdout = logger

    dataset_paths = args.dataset or list(DEFAULT_DATASETS)
    datasets = DatasetCollection(dataset_paths)
    userscores = h5py.File(args.userscore, "r") if args.userscore else None
    try:
        train_keys, test_keys = load_split(args.split, args.split_id)
        print(args)
        print(f"Device: {device}; train videos: {len(train_keys)}; test videos: {len(test_keys)}")
        model = ISCRLPolicy(
            input_dim=args.input_dim,
            state_dim=args.hidden_dim,
            invariant_dim=args.invariant_dim,
            projector_hidden_dim=args.projector_hidden_dim,
            attention_dropout=args.attention_dropout,
        ).to(device)
        aim = AIMController(
            intervention_min=args.intervention_min,
            intervention_max=args.intervention_max,
            penalty=args.intervention_penalty,
            increase_factor=args.intervention_increase,
            decrease_factor=args.intervention_decrease,
            base_clip=args.base_clip,
            baseline_decay=args.baseline_decay,
        )
        start_epoch = 0
        optimizer_state = None
        warmup_history: List[float] = []
        if args.resume:
            checkpoint = torch.load(args.resume, map_location=device)
            state_dict = checkpoint.get("state_dict", checkpoint)
            state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
            model.load_state_dict(state_dict)
            if isinstance(checkpoint, dict):
                aim.load_state_dict(checkpoint.get("aim"))
                start_epoch = int(checkpoint.get("rl_epoch", 0))
                warmup_history = list(checkpoint.get("warmup_history", []))
                optimizer_state = checkpoint.get("optimizer")
            print(f"Loaded checkpoint {args.resume!r} at RL epoch {start_epoch}")

        if args.evaluate:
            evaluate(model, datasets, userscores, test_keys, args, device)
            return

        started = time.time()
        if not args.resume:
            warmup_history = contrastive_warmup(model, datasets, train_keys, args, device)
        rewards, optimizer = train_policy(
            model,
            datasets,
            train_keys,
            args,
            device,
            aim,
            start_epoch=start_epoch,
            optimizer_state=optimizer_state,
        )
        write_json(rewards, os.path.join(args.save_dir, "rewards.json"))
        checkpoint_path = Path(args.save_dir) / (
            f"{args.metric}_iscrl_rl{args.rl_epochs}_split{args.split_id}.pth.tar"
        )
        save_checkpoint(
            checkpoint_state(model, optimizer, aim, args, warmup_history),
            str(checkpoint_path),
        )
        print(f"Model saved to {checkpoint_path}")
        evaluate(model, datasets, userscores, test_keys, args, device)
        elapsed = datetime.timedelta(seconds=round(time.time() - started))
        print(f"Finished in {elapsed}")
    finally:
        datasets.close()
        if userscores is not None:
            userscores.close()
        sys.stdout = original_stdout
        logger.close()


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
