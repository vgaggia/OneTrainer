from modules.model.LongCatImageModel import LongCatImageModel
from modules.modelSaver.GenericLoRAModelSaver import make_lora_model_saver
from modules.modelSaver.longcatImage.LongCatImageLoRASaver import LongCatImageLoRASaver
from modules.util.enum.ModelType import ModelType

LongCatImageLoRAModelSaver = make_lora_model_saver(
    ModelType.LONGCAT_IMAGE_EDIT,
    model_class=LongCatImageModel,
    lora_saver_class=LongCatImageLoRASaver,
    embedding_saver_class=None,
)
