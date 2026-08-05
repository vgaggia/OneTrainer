from modules.model.LongCatImageModel import LongCatImageModel
from modules.modelSaver.GenericFineTuneModelSaver import make_fine_tune_model_saver
from modules.modelSaver.longcatImage.LongCatImageModelSaver import LongCatImageModelSaver
from modules.util.enum.ModelType import ModelType

LongCatImageFineTuneModelSaver = make_fine_tune_model_saver(
    ModelType.LONGCAT_IMAGE_EDIT,
    model_class=LongCatImageModel,
    model_saver_class=LongCatImageModelSaver,
    embedding_saver_class=None,
)
