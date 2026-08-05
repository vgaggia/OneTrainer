from modules.model.LongCatImageModel import LongCatImageModel
from modules.modelLoader.GenericFineTuneModelLoader import make_fine_tune_model_loader
from modules.modelLoader.longcatImage.LongCatImageModelLoader import LongCatImageModelLoader
from modules.util.enum.ModelType import ModelType

LongCatImageFineTuneModelLoader = make_fine_tune_model_loader(
    model_spec_map={ModelType.LONGCAT_IMAGE_EDIT: "resources/sd_model_spec/longcat-image-edit.json"},
    model_class=LongCatImageModel,
    model_loader_class=LongCatImageModelLoader,
    embedding_loader_class=None,
)
