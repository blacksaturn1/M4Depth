"""
----------------------------------------------------------------------------------------
Copyright (c) 2022 - Michael Fonder, University of Liège (ULiège), Belgium.

This program is free software: you can redistribute it and/or modify it under the terms
of the GNU Affero General Public License as published by the Free Software Foundation,
either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU Affero General Public License for more details.
You should have received a copy of the GNU Affero General Public License along with this
program. If not, see < [ https://www.gnu.org/licenses/ | https://www.gnu.org/licenses/ ] >.
----------------------------------------------------------------------------------------
"""

import os
import argparse
from m4depth_options import M4DepthOptions

cmdline = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
model_opts = M4DepthOptions(cmdline)
cmd, test_args = cmdline.parse_known_args()
if cmd.mode == 'eval':
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import tensorflow as tf
from tensorflow import keras
import numpy as np
import dataloaders as dl
from callbacks import *
from m4depth_network import *
from metrics import *
import time

if __name__ == '__main__':

    cmdline = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    model_opts = M4DepthOptions(cmdline)
    cmd, test_args = cmdline.parse_known_args()

    # configure tensorflow gpus
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass

    enable_validation = cmd.enable_validation
    try:
        # Manage GPU memory to be able to run the validation step in parallel on the same GPU
        if cmd.mode == "validation":
            print('limit memory')
            tf.config.set_logical_device_configuration(physical_devices[0],
                                                       [tf.config.LogicalDeviceConfiguration(memory_limit=1200)])
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        print("GPUs initialization failed")
        enable_validation = False
        pass

    working_dir = os.getcwd()
    print("The current working directory is : %s" % working_dir)

    chosen_dataloader = dl.get_loader(cmd.dataset)

    seq_len = cmd.seq_len
    nbre_levels = cmd.arch_depth
    ckpt_dir = cmd.ckpt_dir

    if cmd.mode == 'train' or cmd.mode == 'finetune':

        print("Training on %s" % cmd.dataset)
        tf.random.set_seed(42)
        chosen_dataloader.get_dataset("train", model_opts.dataloader_settings, batch_size=cmd.batch_size)
        data = chosen_dataloader.dataset

        model = M4Depth(depth_type=chosen_dataloader.depth_type,
                        nbre_levels=nbre_levels,
                        ablation_settings=model_opts.ablation_settings,
                        is_training=True)

        # Initialize callbacks
        tensorboard_cbk = keras.callbacks.TensorBoard(
            log_dir=cmd.log_dir, histogram_freq=1200, write_graph=True,
            write_images=False, update_freq=1200,
            profile_batch=0, embeddings_freq=0, embeddings_metadata=None)
        model_checkpoint_cbk = CustomCheckpointCallback(os.path.join(ckpt_dir,"train"), resume_training=True)

        opt = tf.keras.optimizers.Adam(learning_rate=0.0001)

        model.compile(optimizer=opt, metrics=[RootMeanSquaredLogError()])

        if enable_validation:
            val_cbk = [CustomKittiValidationCallback(cmd, args=test_args)]
        else:
            val_cbk = []

        # Adapt number of steps depending on desired usecase
        if cmd.mode == 'finetune':
            nbre_epochs = model_checkpoint_cbk.resume_epoch + (20000 // chosen_dataloader.length)
        else:
            nbre_epochs = (220000 // chosen_dataloader.length)
        print("Training for %d epochs" % nbre_epochs)
        # nbre_epochs = 6 # For testing purposes, limit to 2 epochs
        print("Updated training for %d epochs" % nbre_epochs)
        
        model.fit(data, epochs= nbre_epochs + 1,
                  initial_epoch=model_checkpoint_cbk.resume_epoch,
                  callbacks=[tensorboard_cbk, model_checkpoint_cbk] + val_cbk)

    elif cmd.mode == 'eval' or cmd.mode == 'validation':

        if cmd.mode=="validation":
            weights_dir = os.path.join(ckpt_dir,"train")
        else:
            weights_dir = os.path.join(ckpt_dir,"best")

        print("Evaluating on %s" % cmd.dataset)
        chosen_dataloader.get_dataset("eval", model_opts.dataloader_settings, batch_size=1)
        data = chosen_dataloader.dataset

        tb_callback = tf.keras.callbacks.TensorBoard(log_dir=cmd.log_dir, profile_batch='10, 25')

        model = M4Depth(nbre_levels=nbre_levels, ablation_settings=model_opts.ablation_settings)

        model_checkpoint_cbk = CustomCheckpointCallback(weights_dir, resume_training=True)
        model.compile(metrics=[AbsRelError(),
                               SqRelError(),
                               RootMeanSquaredError(),
                               RootMeanSquaredLogError(),
                               ThresholdRelError(1), ThresholdRelError(2), ThresholdRelError(3)])

        metrics = model.evaluate(data, callbacks=[model_checkpoint_cbk])

        # Keep track of the computed performance
        if cmd.mode == 'validation':
            manager = BestCheckpointManager(os.path.join(ckpt_dir,"train"), os.path.join(ckpt_dir,"best"), keep_top_n=cmd.keep_top_n)
            perfs = {"abs_rel": [metrics[0]], "sq_rel": [metrics[1]], "rmse": [metrics[2]], "rmsel": [metrics[3]],
                     "a1": [metrics[4]], "a2": [metrics[5]], "a3": [metrics[6]]}
            manager.update_backup(perfs)
            string = ''
            for perf in metrics:
                string += format(perf, '.4f') + "\t\t"
            with open(os.path.join(*[ckpt_dir, "validation-perfs.txt"]), 'a') as file:
                file.write(string + '\n')
        else:
            np.savetxt(os.path.join(*[ckpt_dir, "perfs-" + cmd.dataset + ".txt"]), metrics, fmt='%.18e', delimiter='\t',
                       newline='\n')

    elif cmd.mode == "predict":
        # chosen_dataloader.get_dataset("predict", model_opts.dataloader_settings, batch_size=1)
        indices = [1,2, 10, 19, 42, 150, 250, 500, 1000, 1500, 3000]  # examples
        chosen_dataloader.get_dataset("predict", model_opts.dataloader_settings,
                              batch_size=1, indices=indices)
        data = chosen_dataloader.dataset

        model = M4Depth(nbre_levels=nbre_levels, ablation_settings=model_opts.ablation_settings)
        model.compile()
        # model_checkpoint_cbk = CustomCheckpointCallback(os.path.join(ckpt_dir, "best"), resume_training=True)
        model_checkpoint_cbk = CustomCheckpointCallback(ckpt_dir, resume_training=True)
        first_sample = data.take(1)
        model.predict(first_sample, callbacks=[model_checkpoint_cbk])

        is_first_run = True

        # Set up directory to save RGB input images
        save_dir = cmd.save_dir if hasattr(cmd, 'save_dir') and cmd.save_dir is not None else os.path.join(ckpt_dir, "predictions")
        os.makedirs(save_dir, exist_ok=True)

        # Do what you want with the outputs
        for i, sample in enumerate(data):
            if not is_first_run and sample["new_traj"]:
                print("Skipping sample %d as it starts a new trajectory." % i)
                continue  # Skip to next sample if new trajectory starts
            if is_first_run:
                is_first_run = False
                continue
            # if i>10:
            #     break  # For testing purposes, limit to first 10 samples
            

            est = model([[sample], sample["camera"]]) # Run network to get estimates
            d_est = est["depth"][0, :, :, :]        # Estimate : [h,w,1] matrix with depth in meter
            d_gt = sample['depth'][0, :, :, :]      # Ground truth : [h,w,1] matrix with depth in meter
            i_rgb = sample['RGB_im'][0, :, :, :]    # RGB image : [h,w,3] matrix with rgb channels ranging between 0 and 1
            print("Prediction %d done" % i)

            # Save RGB input image to disk as PNG
            rgb_uint8 = tf.image.convert_image_dtype(i_rgb, dtype=tf.uint8, saturate=True)
            img_bytes = tf.io.encode_png(rgb_uint8)
            img_path = os.path.join(save_dir, f"rgb_{i:06d}.png")
            tf.io.write_file(img_path, img_bytes)

            # Save estimated and ground-truth depth as 16-bit PNGs
            max_depth = 80.0  # meters; used for visualization scaling

            def _save_depth(depth_tensor, filename):
                depth_clipped = tf.clip_by_value(depth_tensor, 0.0, max_depth)
                depth_norm = depth_clipped / max_depth
                depth_uint16 = tf.image.convert_image_dtype(depth_norm, dtype=tf.uint16, saturate=True)
                depth_bytes = tf.io.encode_png(depth_uint16)
                depth_path = os.path.join(save_dir, filename)
                tf.io.write_file(depth_path, depth_bytes)

            _save_depth(d_est, f"depth_est_{i:06d}.png")
            _save_depth(d_gt, f"depth_gt_{i:06d}.png")

        # Combine saved images into grids for visualization.
        # Each column corresponds to a sample; for every sample we stack:
        # RGB, estimated depth, ground-truth depth (3 rows per sample),
        # arranged in 10 columns.

        from PIL import Image
        import math

        # Collect indices for which RGB images were saved (matches filename indices)
        saved_indices = [idx for idx in range(0, i + 1)
                         if os.path.exists(os.path.join(save_dir, f"rgb_{idx:06d}.png"))]

        if saved_indices:
            # Use the first saved RGB image to get thumbnail size
            first_rgb_path = os.path.join(save_dir, f"rgb_{saved_indices[0]:06d}.png")
            with Image.open(first_rgb_path) as im0:
                thumb_width, thumb_height = im0.size

            num_images = len(saved_indices)
            grid_cols = 10
            group_rows = math.ceil(num_images / grid_cols)
            grid_rows = group_rows * 3  # RGB, depth_est, depth_gt per sample

            grid_image = Image.new('RGB', (grid_cols * thumb_width,
                                           grid_rows * thumb_height))

            for pos, idx in enumerate(saved_indices):
                col = pos % grid_cols
                group = pos // grid_cols

                # RGB row
                rgb_path = os.path.join(save_dir, f"rgb_{idx:06d}.png")
                with Image.open(rgb_path) as img_rgb:
                    img_rgb = img_rgb.convert('RGB')
                    grid_image.paste(img_rgb,
                                     (col * thumb_width,
                                      (group * 3 + 0) * thumb_height))

                # Estimated depth row
                d_est_path = os.path.join(save_dir, f"depth_est_{idx:06d}.png")
                if os.path.exists(d_est_path):
                    with Image.open(d_est_path) as img_de:
                        img_de = img_de.convert('L')
                        grid_image.paste(img_de.convert('RGB'),
                                         (col * thumb_width,
                                          (group * 3 + 1) * thumb_height))

                # Ground-truth depth row
                d_gt_path = os.path.join(save_dir, f"depth_gt_{idx:06d}.png")
                if os.path.exists(d_gt_path):
                    with Image.open(d_gt_path) as img_dg:
                        img_dg = img_dg.convert('L')
                        grid_image.paste(img_dg.convert('RGB'),
                                         (col * thumb_width,
                                          (group * 3 + 2) * thumb_height))

            grid_image_path = os.path.join(save_dir, "rgb_depth_grid.png")
            grid_image.save(grid_image_path)
            print(f"Saved RGB+depth image grid to {grid_image_path}")


