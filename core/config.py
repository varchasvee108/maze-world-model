from dataclasses import dataclass
import tomllib
from pathlib import Path


@dataclass
class ProjectConfig:
    name: str
    version: str
    experiment_name: str


@dataclass
class InputDataConfig:
    raw_dir: str
    split_dir: str
    batch_size: int
    num_workers: int
    image_size: tuple[int]
    seed: int


@dataclass
class TrainingConfig:
    lr: float
    max_steps: int
    warmup_steps: int
    betas: tuple[float]
    weight_decay: float
    gradient_clipping: float
    eval_interval: int
    log_interval: int
    scheduler: str


@dataclass
class ModelConfig:
    num_layers: int
    n_embd: int
    num_heads: int
    dropout: float
    dim_ratio: int
    patch_size: int
    max_sequence_length: int


@dataclass
class LoggingConfig:
    use_wandb: bool
    assets_dir: str
    project_name: str


@dataclass
class Config:
    project: ProjectConfig
    input_data: InputDataConfig
    training: TrainingConfig
    model: ModelConfig
    logging: LoggingConfig

    @classmethod
    def load(cls, config_path: str = "config/base.toml") -> "Config":
        path_obj = Path(config_path)

        if not path_obj.exists():
            raise FileNotFoundError(f"Config file not found at {path_obj}")
        if not path_obj.is_file():
            raise FileNotFoundError(f"Path is not a file {path_obj}")

        with open(path_obj, "rb") as f:
            data = tomllib.load(f)

        input_data_dict = data["input_data"]
        training_dict = data["training"]
        merged_input_data_dict = {
            **input_data_dict,
            "image_size": tuple(input_data_dict["image_size"]),
        }
        merged_training_dict = {**training_dict, "betas": tuple(training_dict["betas"])}

        return cls(
            project=ProjectConfig(**data["project"]),
            input_data=InputDataConfig(**merged_input_data_dict),
            training=TrainingConfig(**merged_training_dict),
            model=ModelConfig(**data["model"]),
            logging=LoggingConfig(**data["logging"]),
        )
