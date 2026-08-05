from modules.model.LongCatImageModel import LongCatImageModel
from modules.modelLoader.GenericLoRAModelLoader import make_lora_model_loader
from modules.modelLoader.longcatImage.LongCatImageLoRALoader import LongCatImageLoRALoader
from modules.modelLoader.longcatImage.LongCatImageModelLoader import LongCatImageModelLoader
from modules.util.enum.ModelType import ModelType

LongCatImageLoRAModelLoader = make_lora_model_loader(
    model_spec_map={ModelType.LONGCAT_IMAGE_EDIT: "resources/sd_model_spec/longcat-image-edit-lora.json"},
    model_class=LongCatImageModel,
    model_loader_class=LongCatImageModelLoader,
    embedding_loader_class=None,
    lora_loader_class=LongCatImageLoRALoader,
)
