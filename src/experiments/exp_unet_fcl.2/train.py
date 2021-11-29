from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning import loggers
from pytorch_lightning.callbacks import early_stopping, model_checkpoint

from model import TorchLightNet
from generator import MyDataset
from data_frame import make_data_frame
from utils import make_yaml
from settings import AVAIL_GPUS, CSV_FOLDER_PATH, LOGGER_EXP_PATH, CHECK_POINTER_EXP_PATH


dataset, df = make_data_frame(CSV_FOLDER_PATH, 'train.csv', 'val.csv', 'test.csv')
train_df = df.get('train')
val_df = df.get('validate')

batch_size = 128
train_full_batch = (train_df.shape[0] // batch_size) * batch_size

s2_input_size = (13, 64, 64)
train_dataset = MyDataset(train_df[:train_full_batch], is_train=False, s2_input_size=s2_input_size)
train_loader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True, shuffle=True,
                          # num_workers=6
                          )

val_dataset = MyDataset(val_df, is_train=False, s2_input_size=s2_input_size)
val_loader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=True, shuffle=False,
                        # num_workers=6
                        )


logger = loggers.TensorBoardLogger(LOGGER_EXP_PATH)
es = early_stopping.EarlyStopping(monitor='val_loss',
                                  mode='min',
                                  patience=100)

check_pointer = model_checkpoint.ModelCheckpoint(dirpath=CHECK_POINTER_EXP_PATH,
                                                 filename='{epoch}-{val_loss:.4f}-{val_acc:.4f}',
                                                 save_top_k=4,
                                                 save_on_train_epoch_end=False,
                                                 monitor='val_loss',
                                                 mode='min')

pl_model = TorchLightNet(lr=5e-3, weight_decay=5e-3)
print(pl_model.configure_optimizers())

make_yaml(pl_model, dataset, CHECK_POINTER_EXP_PATH)

trainer = pl.Trainer(gpus=AVAIL_GPUS,
                     logger=logger,
                     callbacks=[es, check_pointer],
                     max_epochs=350)

trainer.fit(pl_model,
            train_loader, val_loader)
