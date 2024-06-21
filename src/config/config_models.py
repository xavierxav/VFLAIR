from pydantic import BaseModel, Field, model_validator
from typing import Optional, List
from typing_extensions import Annotated, Self

class ItemAccessibleModel(BaseModel):
    def __getitem__(self, key):
        return getattr(self, key)
    
    def __setitem__(self, key, value):
        setattr(self, key, value)
    
    class Config:
        # Allow assignment to private attributes
        allow_mutation = True
        # Allow extra fields to be added dynamically
        extra = 'allow'


class CommunicationConfig(ItemAccessibleModel):
    communication_protocol: Annotated[str, Field(pattern='^(FedBCD_p|FedSGD)$')]
    iteration_per_aggregation: Annotated[int, Field(ge=1)] = 1
    get_communication_size: Optional[bool] = False
    # quant_level: Optional[Annotated[int, Field(ge=0)]] = None
    # vecdim: Optional[Annotated[int, Field(ge=1)]] = None
    # num_update_per_batch: Optional[Annotated[int, Field(ge=1)]] = None
    # smi_thresh: Optional[Annotated[float, Field(ge=0, le=1)]] = None
    # ratio: Optional[Annotated[float, Field(ge=0, le=1)]] = None

class DatasetConfig(ItemAccessibleModel):
    dataset_name: str
    num_classes: Annotated[int, Field(ge=1)] = 10
    data_root: Optional[str] = r'C:\Users\XD278777\Desktop\VFLAIR_light\data\satellite_dataset'
    features_instead : Optional[str] = None
    transform: Optional[bool] = True
    cloud_cover_ranking: Optional[bool] = False

class ModelConfig(ItemAccessibleModel):
    input_dim: Annotated[int, Field(ge=1)]
    output_dim: Annotated[int, Field(ge=1)]
    type: str
    path: Optional[str] = None
    weight_decay: Optional[float] = 0.0

class RuntimeConfig(ItemAccessibleModel):
    device: Annotated[str, Field(pattern='^(cuda|cpu)$')]
    gpu: Annotated[int, Field(ge=0)] = 0
    seed: Annotated[int, Field(ge=0)] = 97
    n_seeds: Annotated[int, Field(ge=1)] = 1
    current_seed: Optional[int] = 97
    save_model: bool = False
    detect_anomaly: Optional[bool] = False

class GlobalModelConfig(ItemAccessibleModel):
    apply_trainable_layer: Annotated[int, Field(ge=0, le=1)] = 1
    model: str
    weight_decay: Optional[float] = 0.0

class Config(ItemAccessibleModel):
    epochs: Annotated[int, Field(ge=1)] = 10
    lr: Annotated[float, Field(ge=0, le=1)] = 0.001
    batch_size: Annotated[int, Field(ge=1)] = 256
    early_stop_threshold: Annotated[int, Field(ge=0)] = 5
    k: Annotated[int, Field(ge=1)] = 2
    communication: CommunicationConfig
    dataset: DatasetConfig
    model_list: List[ModelConfig]
    global_model: GlobalModelConfig
    runtime: RuntimeConfig
    compare_centralized: bool = False
    compare_single: bool = False
    
    @model_validator(mode='after')
    def check_model_list_length(self) -> Self:
        k = self.k
        model_list = self.model_list
        if len(model_list) == k:
            return self
        elif len(model_list) == 1:
            self.model_list = model_list * k
            return self
        else:
            raise ValueError("Length of model_list must be either 1 (all models are the same) or equal to k (the number of parties)")


