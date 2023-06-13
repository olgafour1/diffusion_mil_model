import logging
import time
import gc

import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from scipy.stats import ttest_rel
from tqdm import tqdm
from data_loader import *
from ema import EMA
from model import *
from pretraining.encoder import Encoder as Encoder
from utils import *
from diffusion_utils import *
import os
from utils import EarlyStopper

plt.style.use('ggplot')


class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.csv_file=args.csv_file
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.model_var_type = config.model.var_type
        self.num_timesteps = config.diffusion.timesteps
        self.vis_step = config.diffusion.vis_step
        self.num_figs = config.diffusion.num_figs

        betas = make_beta_schedule(schedule=config.diffusion.beta_schedule, num_timesteps=self.num_timesteps,
                                   start=config.diffusion.beta_start, end=config.diffusion.beta_end)
        betas = self.betas = betas.float().to(self.device)
        self.betas_sqrt = torch.sqrt(betas)
        alphas = 1.0 - betas
        self.alphas = alphas
        self.one_minus_betas_sqrt = torch.sqrt(alphas)
        alphas_cumprod = alphas.cumprod(dim=0)
        self.alphas_bar_sqrt = torch.sqrt(alphas_cumprod)
        self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_cumprod)
        if config.diffusion.beta_schedule == "cosine":
            self.one_minus_alphas_bar_sqrt *= 0.9999  # avoid division by 0 for 1/sqrt(alpha_bar_t) during inference
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        self.alphas_cumprod_prev = alphas_cumprod_prev
        self.posterior_mean_coeff_1 = (
                betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.posterior_mean_coeff_2 = (
                torch.sqrt(alphas) * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)
        )
        posterior_variance = (
                betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.posterior_variance = posterior_variance
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
            # torch.cat(
            # [posterior_variance[1:2], betas[1:]], dim=0).log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

        # initial prediction model as guided condition
        self.tuned_scale_T = None
        if config.diffusion.apply_aux_cls:

            self.cond_pred_model = Encoder(k=3, n_classes=2, weight_decay=None).to(self.device)

            self.aux_cost_function = nn.CrossEntropyLoss()


    # Compute guiding prediction as diffusion condition
    def compute_guiding_prediction(self, x):
        """
        Compute y_0_hat, to be used as the Gaussian mean at time step T.
        """
        y_pred = self.cond_pred_model(x)
        return y_pred

    def evaluate_guidance_model(self, dataset_loader):
        """
        Evaluate guidance model by reporting train or test set accuracy.
        """
        y_acc_list = []
        for step, feature_label_set in tqdm(enumerate(dataset_loader)):

            features, indices, values, y_labels_batch = feature_label_set
            y_labels_batch = y_labels_batch.reshape(-1, 1)
            y_pred_prob = self.compute_guiding_prediction([
                torch.squeeze(features).to(self.device),
                torch.squeeze(indices).to(self.device),
                torch.squeeze(values).to(self.device)
            ]) # (batch_size, n_classes)
            y_pred_label = torch.argmax(y_pred_prob, 1, keepdim=True).cpu().detach().numpy()  # (batch_size, 1)
            y_labels_batch = y_labels_batch.cpu().detach().numpy()
            y_acc = y_pred_label == y_labels_batch  # (batch_size, 1)
            if len(y_acc_list) == 0:
                y_acc_list = y_acc
            else:
                y_acc_list = np.concatenate([y_acc_list, y_acc], axis=0)
        y_acc_all = np.mean(y_acc_list)
        return y_acc_all

    def nonlinear_guidance_model_train_step(self, x_batch, y_batch, aux_optimizer):
        """
        One optimization step of the non-linear guidance model that predicts y_0_hat.
        """
        #aux_optimizer.zero_grad()

        pred = self.compute_guiding_prediction(x_batch)
        if  isinstance(pred, tuple):
            aux_cost = self.aux_cost_function(pred[0], y_batch) + pred[1]
        else:
            aux_cost = self.aux_cost_function(pred, y_batch)
        # update non-linear guidance model
        aux_optimizer.zero_grad()
        aux_cost.backward()
        aux_optimizer.step()
        return aux_cost.cpu().item()

    def nonlinear_guidance_model_val_step(self, x_batch, y_batch):
        """
        One optimization step of the non-linear guidance model that predicts y_0_hat.
        """
        # aux_optimizer.zero_grad()

        pred = self.compute_guiding_prediction(x_batch)
        if isinstance(pred, tuple):
            aux_cost = self.aux_cost_function(pred[0], y_batch) + pred[1]
            return aux_cost.cpu().item(), pred[0]
        else:
            aux_cost = self.aux_cost_function(pred, y_batch)
            return aux_cost.cpu().item(), pred

    def train(self):

        args = self.args
        config = self.config
        tb_logger = self.config.tb_logger

        references = pd.read_csv( self.csv_file)

        train_bags = references["train"].apply(lambda x: os.path.join(self.config.data.feature_path, x + ".h5")).values.tolist()

        def func_val(x):
            value = None
            if isinstance(x, str):
                value = os.path.join(self.config.data.feature_path, x + ".h5")
            return value

        val_bags = references.apply(lambda row: func_val(row.val), axis=1).dropna().values.tolist()

        test_bags = references.apply(lambda row: func_val(row.test), axis=1).dropna().values.tolist()

        train_dataloader = Whole_Slide_Bag(train_bags, label_file=self.config.data.label_file, dataset='camelyon')
        val_dataloader = Whole_Slide_Bag(val_bags, label_file=self.config.data.label_file, dataset='camelyon')
        test_dataloader = Whole_Slide_Bag(test_bags, label_file=self.config.data.label_file, dataset='camelyon')

        train_dataloader = torch.utils.data.DataLoader(train_dataloader, batch_size=1, shuffle=True)
        val_dataloader = torch.utils.data.DataLoader(val_dataloader, batch_size=1, shuffle=True)
        test_dataloader = torch.utils.data.DataLoader(test_dataloader, batch_size=1, shuffle=False)

        model = ConditionalModel(config, guidance=config.diffusion.include_guidance)
        model = model.to(self.device)


        if self.config.model.ema:
            ema_helper = EMA(mu=self.config.model.ema_rate)
            ema_helper.register(model)
        else:
            ema_helper = None

        optimizer = get_optimizer(self.config.optim, model.parameters())
        criterion = nn.CrossEntropyLoss()

        if config.diffusion.apply_aux_cls:
            # aux_optimizer = get_optimizer(self.config.aux_optim,
            #                               self.cond_pred_model.parameters())
            aux_optimizer = torch.optim.Adam(self.cond_pred_model.parameters(), lr=0.0002,  weight_decay=0, eps = 0.0000001, amsgrad = False)
            aux_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(aux_optimizer, mode='min',verbose=True,
                                                                   factor=0.2, patience=10, threshold=0.0001,
                                                                   threshold_mode='abs')


        pretrain_start_time = time.time()
        min_valid_loss = np.inf
        early_stopper = EarlyStopper(patience=20)
        # val_acc = BinaryAccuracy()
        # val_acc = val_acc.to(self.device)
        for epoch in range(config.diffusion.aux_cls.n_pretrain_epochs):
            train_loss = 0.
            val_loss = 0
            self.cond_pred_model.train()
            for step, (train_features, train_indices, train_values, bag_label) in enumerate(train_dataloader):

                y_one_hot_batch, y_logits_batch = cast_bag_label_to_one_hot_and_prototype(bag_label,
                                                                                      config)
                #print(bag_label,y_one_hot_batch, y_logits_batch)

                aux_cost = self.nonlinear_guidance_model_train_step([torch.squeeze(train_features).to(self.device),
                                                            torch.squeeze(train_indices).to(self.device) ,
                                                            torch.squeeze(train_values).to(self.device)],
                                                            y_one_hot_batch.to(self.device),
                                                            aux_optimizer)

                train_loss += aux_cost

            train_loss /= len(train_dataloader)
            if epoch % config.diffusion.aux_cls.logging_interval == 0:
                    logging.info(
                        "EPOCH: {}, guidance auxiliary classifier pre-training loss: {}".format(epoch,train_loss)
                                )

            self.cond_pred_model.eval()
            with torch.no_grad():
                for step, (val_features, val_indices, val_values, bag_label) in enumerate(val_dataloader):
                        y_one_hot_batch, y_logits_batch = cast_bag_label_to_one_hot_and_prototype(bag_label,
                                                                                              config)
                        aux_cost, y_pred_label = self.nonlinear_guidance_model_val_step([torch.squeeze(val_features).to(self.device),
                                                                         torch.squeeze(val_indices).to(self.device),
                                                                         torch.squeeze(val_values).to(self.device)],
                                                                         y_one_hot_batch.to(self.device))

                        val_loss += aux_cost
                        # y_pred_label = y_pred_label.to(self.device)
                        # bag_label = bag_label.to(self.device)
                        # val_acc.update( torch.squeeze(y_pred_label,0),  torch.squeeze(bag_label,0))


            val_loss /= len(val_dataloader)
            aux_scheduler.step(val_loss)
            # print(f'Validation Accuracy ---> {val_acc.compute():.6f}')
            # val_acc.reset()
            if min_valid_loss > val_loss:
                print(f'Validation Loss Decreased ({min_valid_loss:.6f}--->{val_loss:.6f}) \t Saving The Model')
                min_valid_loss = val_loss
                # Saving State Dict
                aux_states = [
                    self.cond_pred_model.state_dict(),
                    aux_optimizer.state_dict(),
                ]
                torch.save(aux_states, os.path.join(self.args.log_path, "aux_ckpt.pth"))
            if early_stopper.early_stop(val_loss):
                break

        pretrain_end_time = time.time()
        logging.info("\nPre-training of guidance auxiliary classifier took {:.4f} minutes.\n".format(
                (pretrain_end_time - pretrain_start_time) / 60))

        y_acc_aux_model = self.evaluate_guidance_model(test_dataloader)
        logging.info("\nAfter pre-training, guidance classifier accuracy on the test set is {:.8f}.\n".format(
            y_acc_aux_model))

        if not self.args.train_guidance_only:
            start_epoch, step = 0, 0
            if self.args.resume_training:
                states = torch.load(os.path.join(self.args.log_path, "ckpt.pth"),
                                    map_location=self.device)
                model.load_state_dict(states[0])

                states[1]["param_groups"][0]["eps"] = self.config.optim.eps
                optimizer.load_state_dict(states[1])
                start_epoch = states[2]
                step = states[3]
                if self.config.model.ema:
                    ema_helper.load_state_dict(states[4])
                # load auxiliary model
                if config.diffusion.apply_aux_cls and (
                        hasattr(config.diffusion, "trained_aux_cls_ckpt_path") is False) and (
                        hasattr(config.diffusion, "trained_aux_cls_log_path") is False):
                    aux_states = torch.load(os.path.join(self.args.log_path, "aux_ckpt.pth"),
                                            map_location=self.device)
                    self.cond_pred_model.load_state_dict(aux_states[0])
                    aux_optimizer.load_state_dict(aux_states[1])

            max_accuracy = 0.0
            if config.diffusion.noise_prior:  # apply 0 instead of f_phi(x) as prior mean
                logging.info("Prior distribution at timestep T has a mean of 0.")
            if args.add_ce_loss:
                logging.info("Apply cross entropy as an auxiliary loss during training.")

            for epoch in range(start_epoch, self.config.training.n_epochs):
                data_start = time.time()
                data_time = 0

                for step, (train_features, train_indices, train_values, bag_label) in enumerate(train_dataloader):
                    if config.optim.lr_schedule:
                        adjust_learning_rate(optimizer, step / len(train_dataloader) + epoch, config)

                    y_one_hot_batch, y_logits_batch = cast_bag_label_to_one_hot_and_prototype(bag_label,
                                                                                              config)
                    n = train_features.size(0)

                    model.train()
                    self.cond_pred_model.eval()
                    step += 1

                    # antithetic sampling
                    t = torch.randint(
                        low=0, high=self.num_timesteps, size=(n // 2 + 1,)
                    ).to(self.device)

                    t = torch.cat([t, self.num_timesteps - 1 - t], dim=0)[:n]

                    y_0_hat_batch  = self.compute_guiding_prediction([
                    torch.squeeze(train_features).to(self.device),
                    torch.squeeze(train_indices).to(self.device),
                    torch.squeeze(train_values).to(self.device)
                    ])

                    y_T_mean = y_0_hat_batch
                    if config.diffusion.noise_prior:  # apply 0 instead of f_phi(x) as prior mean
                        y_T_mean = torch.zeros(y_0_hat_batch.shape).to(y_0_hat_batch.device)

                    y_0_batch = y_one_hot_batch.to(self.device)
                    e = torch.randn_like(y_0_batch).to(y_0_batch.device)
                    y_t_batch = q_sample(y_0_batch, y_T_mean,
                                         self.alphas_bar_sqrt, self.one_minus_alphas_bar_sqrt, t, noise=e)

                    output = model( torch.squeeze(train_features).to(self.device), y_t_batch.to(self.device), t, y_0_hat_batch)

                    loss = (e - output).square().mean()  # use the same noise sample e during training to compute loss
                    loss0 = torch.tensor([0])
                    ce_loss= torch.tensor([0])

                    if args.add_ce_loss:

                        y_0_reparam_batch = y_0_reparam(model, torch.squeeze(train_features).to(self.device), y_t_batch, y_0_hat_batch, y_T_mean, t,
                                                        self.one_minus_alphas_bar_sqrt)

                        raw_prob_batch = -(y_0_reparam_batch - 1) ** 2

                        bag_label= torch.squeeze(bag_label.long(), dim=1).to(self.device)

                        loss0 = criterion(raw_prob_batch, bag_label)
                        loss += config.training.lambda_ce * loss0

                    if not tb_logger is None:
                        tb_logger.add_scalar("loss", loss, global_step=step)

                    if step % self.config.training.logging_freq == 0 or step == 1:
                        logging.info(
                            (
                                    f"epoch: {epoch}, step: {step}, CE loss: {loss0.item()}, "
                                    f"Noise Estimation loss: {loss.item()}, " +
                                    f"data time: {data_time / (step + 1)}"
                            )
                        )

                    # optimize diffusion model that predicts eps_theta
                    optimizer.zero_grad()
                    loss.backward()
                    try:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), config.optim.grad_clip
                        )
                    except Exception:
                        pass
                    optimizer.step()
                    if self.config.model.ema:
                        ema_helper.update(model)

                logging.info(
                    (f"epoch: {epoch}, step: {step}, CE loss: {loss0.item()}, Noise Estimation loss: {loss.item()}, " +
                     f"data time: {data_time / (step + 1)}")
                )

                # save diffusion model
                if step % self.config.training.snapshot_freq == 0 or step == 1:
                    states = [
                        model.state_dict(),
                        optimizer.state_dict(),
                        epoch,
                        step,
                    ]
                    if self.config.model.ema:
                        states.append(ema_helper.state_dict())

                    if step > 1:
                        torch.save(
                                states,
                                os.path.join(self.args.log_path, "ckpt_{}.pth".format(step)),
                            )
                        # save current states
                        torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))

                    # save auxiliary model
                    if config.diffusion.apply_aux_cls and config.diffusion.aux_cls.joint_train:
                        aux_states = [
                            self.cond_pred_model.state_dict(),
                            aux_optimizer.state_dict(),
                        ]
                        if step > 1:  # skip saving the initial ckpt
                            torch.save(
                                aux_states,
                                os.path.join(self.args.log_path, "aux_ckpt_{}.pth".format(step)),
                            )
                        torch.save(aux_states, os.path.join(self.args.log_path, "aux_ckpt.pth"))

                if epoch % self.config.training.validation_freq == 0 or epoch + 1 == self.config.training.n_epochs:
                        model.eval()
                        self.cond_pred_model.eval()
                        acc_avg = 0.
                        for test_batch_idx, (val_features, val_indices,val_values,target ) in enumerate(test_dataloader):

                            val_features= torch.squeeze(val_features).to(self.device)
                            val_indices= torch.squeeze(val_indices).to(self.device)
                            val_values = torch.squeeze(val_values).to(self.device)
                            target = target.to(self.device)
                            # target_vec = nn.functional.one_hot([]).float().to(self.device)
                            with torch.no_grad():
                                target_pred = self.compute_guiding_prediction([val_features, val_indices, val_values])
                                # prior mean at timestep T
                                y_T_mean = target_pred
                                if config.diffusion.noise_prior:  # apply 0 instead of f_phi(x) as prior mean
                                    y_T_mean = torch.zeros(target_pred.shape).to(target_pred.device)
                                if not config.diffusion.noise_prior:  # apply f_phi(x) instead of 0 as prior mean
                                    target_pred = self.compute_guiding_prediction([val_features, val_indices, val_values])
                                label_t_0 = p_sample_loop(model, val_features, target_pred, y_T_mean,
                                                          self.num_timesteps, self.alphas,
                                                          self.one_minus_alphas_bar_sqrt,
                                                          only_last_sample=True)
                                acc_avg += accuracy(label_t_0.detach().cpu(), target.cpu())[0].item()
                        acc_avg /= (test_batch_idx + 1)
                        if acc_avg > max_accuracy:
                            logging.info("Update best accuracy at Epoch {}.".format(epoch))
                            torch.save(states, os.path.join(self.args.log_path, "ckpt_best.pth"))
                        max_accuracy = max(max_accuracy, acc_avg)
                        if not tb_logger is None:
                            tb_logger.add_scalar('accuracy', acc_avg, global_step=step)
                        logging.info(
                            (
                                    f"epoch: {epoch}, step: {test_batch_idx}, " +
                                    f"Average accuracy: {acc_avg}, " +
                                    f"Max accuracy: {max_accuracy:.2f}%"
                            )
                        )
            # save the model after training is finished
            states = [
                            model.state_dict(),
                            optimizer.state_dict(),
                            epoch,
                            step,
                        ]
            if self.config.model.ema:
                states.append(ema_helper.state_dict())
            torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))
            # save auxiliary model after training is finished
            if config.diffusion.apply_aux_cls and config.diffusion.aux_cls.joint_train:
                aux_states = [
                    self.cond_pred_model.state_dict(),
                    aux_optimizer.state_dict(),
                ]
                torch.save(aux_states, os.path.join(self.args.log_path, "aux_ckpt.pth"))
                # report training set accuracy if applied joint training
                y_acc_aux_model = self.evaluate_guidance_model(train_dataloader)
                logging.info("After joint-training, guidance classifier accuracy on the training set is {:.8f}.".format(
                    y_acc_aux_model))
                # report test set accuracy if applied joint training
                y_acc_aux_model = self.evaluate_guidance_model(val_dataloader)
                logging.info("After joint-training, guidance classifier accuracy on the test set is {:.8f}.".format(
                    y_acc_aux_model))

    def test(self):
        """
        Evaluate model performance on image classification tasks.
        """

        def compute_val_before_softmax(gen_y_0_val):
            """
            Compute raw value before applying Softmax function to obtain prediction in probability scale.
            Corresponding to the part inside the Softmax function of Eq. (10) in paper.
            """
            # TODO: add other ways of computing such raw prob value
            raw_prob_val = -(gen_y_0_val - 1) ** 2
            return raw_prob_val

        #####################################################################################################
        ########################## local functions within the class function scope ##########################
        def compute_and_store_cls_metrics(config, y_labels_batch, generated_y, batch_size, num_t):
            """
            generated_y: y_t in logit, has a shape of (batch_size x n_samples, n_classes)

            For each instance, compute probabilities of prediction of each label, majority voted label and
                its correctness, as well as accuracy of all samples given the instance.
            """
            current_t = self.num_timesteps + 1 - num_t
            gen_y_all_class_raw_probs = generated_y.reshape(batch_size,
                                                            config.testing.n_samples,
                                                            config.data.num_classes).cpu()  # .numpy()
            # compute softmax probabilities of all classes for each sample
            raw_prob_val = compute_val_before_softmax(gen_y_all_class_raw_probs)
            gen_y_all_class_probs = torch.softmax(raw_prob_val / self.tuned_scale_T,
                                                  dim=-1)  # (batch_size, n_samples, n_classes)
            # obtain credible interval of probability predictions in each class for the samples given the same x
            low, high = config.testing.PICP_range
            # use raw predicted probability (right before temperature scaling and Softmax) width
            # to construct prediction interval
            CI_y_pred = raw_prob_val.nanquantile(q=torch.tensor([low / 100, high / 100]),
                                                 dim=1).swapaxes(0, 1)  # (batch_size, 2, n_classes)
            # obtain the predicted label with the largest probability for each sample
            gen_y_labels = torch.argmax(gen_y_all_class_probs, 2, keepdim=True)  # (batch_size, n_samples, 1)
            # convert the predicted label to one-hot format
            gen_y_one_hot = torch.zeros_like(gen_y_all_class_probs).scatter_(
                dim=2, index=gen_y_labels,
                src=torch.ones_like(gen_y_labels.float()))  # (batch_size, n_samples, n_classes)
            # compute proportion of each class as the prediction given the same x
            gen_y_label_probs = gen_y_one_hot.sum(1) / config.testing.n_samples  # (batch_size, n_classes)
            gen_y_all_class_mean_prob = gen_y_all_class_probs.mean(1)  # (batch_size, n_classes)
            # obtain the class being predicted the most given the same x
            gen_y_majority_vote = torch.argmax(gen_y_label_probs, 1, keepdim=True)  # (batch_size, 1)
            # compute the proportion of predictions being the correct label for each x
            gen_y_instance_accuracy = (gen_y_labels == y_labels_batch[:, None]).float().mean(1)  # (batch_size, 1)
            # conduct paired two-sided t-test for the two most predicted classes for each instance
            two_most_probable_classes_idx = gen_y_label_probs.argsort(dim=1, descending=True)[:, :2]
            two_most_probable_classes_idx = torch.repeat_interleave(
                two_most_probable_classes_idx[:, None],
                repeats=config.testing.n_samples, dim=1)  # (batch_size, n_samples, 2)
            gen_y_2_class_probs = torch.gather(gen_y_all_class_probs, dim=2,
                                               index=two_most_probable_classes_idx)  # (batch_size, n_samples, 2)

            # make plots to check normality assumptions (differences btw two most probable classes) for the t-test
            # (check https://www.researchgate.net/post/Paired_t-test_and_normality_test_question)

            ttest_pvalues = (ttest_rel(gen_y_2_class_probs[:, :, 0],
                                       gen_y_2_class_probs[:, :, 1],
                                       axis=1, alternative='two-sided')).pvalue  # (batch_size, )
            ttest_reject = (ttest_pvalues < config.testing.ttest_alpha)  # (batch_size, )

            if len(majority_vote_by_batch_list[current_t]) == 0:
                majority_vote_by_batch_list[current_t] = gen_y_majority_vote
            else:
                majority_vote_by_batch_list[current_t] = np.concatenate(
                    [majority_vote_by_batch_list[current_t], gen_y_majority_vote], axis=0)
            if current_t == config.testing.metrics_t:
                gen_y_all_class_probs = gen_y_all_class_probs.reshape(
                    y_labels_batch.shape[0] * config.testing.n_samples, config.data.num_classes)
                if len(label_probs_by_batch_list[current_t]) == 0:
                    all_class_probs_by_batch_list[current_t] = gen_y_all_class_probs
                    label_probs_by_batch_list[current_t] = gen_y_label_probs
                    label_mean_probs_by_batch_list[current_t] = gen_y_all_class_mean_prob
                    instance_accuracy_by_batch_list[current_t] = gen_y_instance_accuracy
                    CI_by_batch_list[current_t] = CI_y_pred
                    ttest_reject_by_batch_list[current_t] = ttest_reject
                else:
                    all_class_probs_by_batch_list[current_t] = np.concatenate(
                        [all_class_probs_by_batch_list[current_t], gen_y_all_class_probs], axis=0)
                    label_probs_by_batch_list[current_t] = np.concatenate(
                        [label_probs_by_batch_list[current_t], gen_y_label_probs], axis=0)
                    label_mean_probs_by_batch_list[current_t] = np.concatenate(
                        [label_mean_probs_by_batch_list[current_t], gen_y_all_class_mean_prob], axis=0)
                    instance_accuracy_by_batch_list[current_t] = np.concatenate(
                        [instance_accuracy_by_batch_list[current_t], gen_y_instance_accuracy], axis=0)
                    CI_by_batch_list[current_t] = np.concatenate(
                        [CI_by_batch_list[current_t], CI_y_pred], axis=0)
                    ttest_reject_by_batch_list[current_t] = np.concatenate(
                        [ttest_reject_by_batch_list[current_t], ttest_reject], axis=0)

        def p_sample_loop_with_eval(model, x, y_0_hat, y_T_mean, n_steps,
                                    alphas, one_minus_alphas_bar_sqrt,
                                    batch_size, config):
            """
            Sample y_{t-1} given y_t on the fly, and evaluate model immediately, to avoid OOM.
            """

            def optional_metric_compute(cur_y, num_t):
                if config.testing.compute_metric_all_steps or \
                        (self.num_timesteps + 1 - num_t == config.testing.metrics_t):
                    compute_and_store_cls_metrics(config, y_labels_batch, cur_y, batch_size, num_t)

            device = next(model.parameters()).device
            z = torch.randn_like(y_T_mean).to(device)  # standard Gaussian
            cur_y = z + y_T_mean  # sampled y_T
            num_t = 1
            optional_metric_compute(cur_y, num_t)
            for i in reversed(range(1, n_steps)):
                y_t = cur_y
                cur_y = p_sample(model, x, y_t, y_0_hat, y_T_mean, i, alphas, one_minus_alphas_bar_sqrt)  # y_{t-1}
                num_t += 1
                optional_metric_compute(cur_y, num_t)
            assert num_t == n_steps
            # obtain y_0 given y_1
            num_t += 1
            y_0 = p_sample_t_1to0(model, x, cur_y, y_0_hat, y_T_mean, one_minus_alphas_bar_sqrt)
            optional_metric_compute(y_0, num_t)

        def p_sample_loop_only_y_0(model, x, y_0_hat, y_T_mean, n_steps,
                                   alphas, one_minus_alphas_bar_sqrt):
            """
            Only compute y_0 -- no metric evaluation.
            """
            device = next(model.parameters()).device
            z = torch.randn_like(y_T_mean).to(device)  # standard Gaussian
            cur_y = z + y_T_mean  # sampled y_T
            num_t = 1
            for i in reversed(range(1, n_steps)):
                y_t = cur_y
                cur_y = p_sample(model, x, y_t, y_0_hat, y_T_mean, i, alphas, one_minus_alphas_bar_sqrt)  # y_{t-1}
                num_t += 1
            assert num_t == n_steps
            y_0 = p_sample_t_1to0(model, x, cur_y, y_0_hat, y_T_mean, one_minus_alphas_bar_sqrt)
            return y_0

        def compute_quantile_metrics(config, CI_all_classes, majority_voted_class, true_y_label):
            """
            CI_all_classes: (n_test, 2, n_classes), contains quantiles at (low, high) in config.testing.PICP_range
                for the predicted probabilities of all classes for each test instance.

            majority_voted_class: (n_test, ) contains whether the majority-voted label is correct or not
                for each test instance.

            true_y_label: (n_test, ) contains true label for each test instance.
            """
            # obtain credible interval width by computing high - low
            CI_width_all_classes = torch.tensor(
                CI_all_classes[:, 1] - CI_all_classes[:, 0]).squeeze()  # (n_test, n_classes)
            # predict by the k-th smallest CI width
            for kth_smallest in range(1, config.data.num_classes + 1):
                pred_by_narrowest_CI_width = torch.kthvalue(
                    CI_width_all_classes, k=kth_smallest, dim=1, keepdim=False).indices.numpy()  # (n_test, )
                # logging.info(pred_by_narrowest_CI_width)  #@#
                narrowest_CI_pred_correctness = (pred_by_narrowest_CI_width == true_y_label)  # (n_test, )
                logging.info(("We predict the label by the class with the {}-th narrowest CI width, \n" +
                              "and obtain a test accuracy of {:.4f}% through the entire test set.").format(
                    kth_smallest, np.mean(narrowest_CI_pred_correctness) * 100))
            # check whether the most predicted class is the correct label for each x
            majority_vote_correctness = (majority_voted_class == true_y_label)  # (n_test, )
            # obtain one-hot label
            true_y_one_hot = cast_label_to_one_hot_and_prototype(torch.tensor(true_y_label), config,
                                                                 return_prototype=False)  # (n_test, n_classes)
            # obtain predicted CI width only for the true class
            CI_width_true_class = (CI_width_all_classes * true_y_one_hot).sum(dim=1, keepdim=True)  # (n_test, 1)
            # sanity check
            nan_idx = torch.arange(true_y_label.shape[0])[CI_width_true_class.flatten().isnan()]
            if nan_idx.shape[0] > 0:
                logging.info(("Sanity check: model prediction contains nan " +
                              "for test instances with index {}.").format(nan_idx.numpy()))
            CI_width_true_class_correct_pred = CI_width_true_class[majority_vote_correctness]  # (n_correct, 1)
            CI_width_true_class_incorrect_pred = CI_width_true_class[~majority_vote_correctness]  # (n_incorrect, 1)
            logging.info(("\n\nWe apply the majority-voted class label as our prediction, and achieve {:.4f}% " +
                          "accuracy through the entire test set.\n" +
                          "Out of {} test instances, we made {} correct predictions, with a " +
                          "mean credible interval width of {:.4f} in predicted probability of the true class; \n" +
                          "the remaining {} instances are classified incorrectly, " +
                          "with a mean CI width of {:.4f}.\n").format(
                np.mean(majority_vote_correctness) * 100,
                true_y_label.shape[0],
                majority_vote_correctness.sum(),
                CI_width_true_class_correct_pred.mean().item(),
                true_y_label.shape[0] - majority_vote_correctness.sum(),
                CI_width_true_class_incorrect_pred.mean().item()))

            maj_vote_acc_by_class = []
            CI_w_cor_pred_by_class = []
            CI_w_incor_pred_by_class = []
            # report metrics within each class
            for c in range(config.data.num_classes):
                maj_vote_cor_class_c = majority_vote_correctness[true_y_label == c]  # (n_class_c, 1)
                CI_width_true_class_c = CI_width_true_class[true_y_label == c]  # (n_class_c, 1)
                CI_w_cor_class_c = CI_width_true_class_c[maj_vote_cor_class_c]  # (n_correct_class_c, 1)
                CI_w_incor_class_c = CI_width_true_class_c[~maj_vote_cor_class_c]  # (n_incorrect_class_c, 1)
                logging.info(("\n\tClass {} ({} total instances, {:.4f}% accuracy):" +
                              "\n\t\t{} correct predictions, mean CI width {:.4f}" +
                              "\n\t\t{} incorrect predictions, mean CI width {:.4f}").format(
                    c, maj_vote_cor_class_c.shape[0], np.mean(maj_vote_cor_class_c) * 100,
                    CI_w_cor_class_c.shape[0], CI_w_cor_class_c.mean().item(),
                    CI_w_incor_class_c.shape[0], CI_w_incor_class_c.mean().item()))
                maj_vote_acc_by_class.append(np.mean(maj_vote_cor_class_c))
                CI_w_cor_pred_by_class.append(CI_w_cor_class_c.mean().item())
                CI_w_incor_pred_by_class.append(CI_w_incor_class_c.mean().item())

            return maj_vote_acc_by_class, CI_w_cor_pred_by_class, CI_w_incor_pred_by_class

        def compute_ttest_metrics(config, ttest_reject, majority_voted_class, true_y_label):
            """
            ttest_reject: (n_test, ) contains whether to reject the paired two-sided t-test between
                the two most probable predicted classes for each test instance.

            majority_voted_class: (n_test, ) contains whether the majority-voted label is correct or not
                for each test instance.

            true_y_label: (n_test, ) contains true label for each test instance.
            """
            # check whether the most predicted class is the correct label for each x
            majority_vote_correctness = (majority_voted_class == true_y_label)  # (n_test, )
            # split test instances into correct and incorrect predictions
            ttest_reject_correct_pred = ttest_reject[majority_vote_correctness]  # (n_correct, )
            ttest_reject_incorrect_pred = ttest_reject[~majority_vote_correctness]  # (n_incorrect, )
            logging.info(("\n\nWe apply the majority-voted class label as our prediction, and achieve {:.4f}% " +
                          "accuracy through the entire test set.\n" +
                          "Out of {} test instances, we made {} correct predictions, with a " +
                          "mean rejection rate of {:.4f}% for the paired two-sided t-test; \n" +
                          "the remaining {} instances are classified incorrectly, " +
                          "with a mean rejection rate of {:.4f}%.\n").format(
                np.mean(majority_vote_correctness) * 100,
                true_y_label.shape[0],
                ttest_reject_correct_pred.shape[0],
                ttest_reject_correct_pred.mean().item() * 100,
                ttest_reject_incorrect_pred.shape[0],
                ttest_reject_incorrect_pred.mean().item() * 100))

            # rejection rate by prediction correctness within each class
            maj_vote_acc_by_class = []
            ttest_rej_cor_pred_by_class = []
            ttest_rej_incor_pred_by_class = []
            for c in range(config.data.num_classes):
                maj_vote_cor_class_c = majority_vote_correctness[true_y_label == c]  # (n_class_c, 1)
                ttest_reject_class_c = ttest_reject[true_y_label == c]  # (n_class_c, 1)
                ttest_rej_cor_class_c = ttest_reject_class_c[maj_vote_cor_class_c]  # (n_correct_class_c, 1)
                ttest_rej_incor_class_c = ttest_reject_class_c[~maj_vote_cor_class_c]  # (n_incorrect_class_c, 1)
                logging.info(("\n\tClass {} ({} total instances, {:.4f}% accuracy):" +
                              "\n\t\t{} correct predictions, mean rejection rate {:.4f}%" +
                              "\n\t\t{} incorrect predictions, mean rejection rate {:.4f}%").format(
                    c, maj_vote_cor_class_c.shape[0], np.mean(maj_vote_cor_class_c) * 100,
                    ttest_rej_cor_class_c.shape[0], ttest_rej_cor_class_c.mean().item() * 100,
                    ttest_rej_incor_class_c.shape[0], ttest_rej_incor_class_c.mean().item() * 100))
                maj_vote_acc_by_class.append(np.mean(maj_vote_cor_class_c))
                ttest_rej_cor_pred_by_class.append(ttest_rej_cor_class_c.mean().item())
                ttest_rej_incor_pred_by_class.append(ttest_rej_incor_class_c.mean().item())

            # split test instances into rejected and not-rejected t-tests
            maj_vote_cor_reject = majority_vote_correctness[ttest_reject]  # (n_reject, )
            maj_vote_cor_not_reject = majority_vote_correctness[~ttest_reject]  # (n_not_reject, )
            logging.info(("\n\nFurthermore, among all test instances, " +
                          "we reject {} t-tests, with a " +
                          "mean accuracy of {:.4f}%; \n" +
                          "the remaining {} t-tests are not rejected, " +
                          "with a mean accuracy of {:.4f}%.\n").format(
                maj_vote_cor_reject.shape[0],
                maj_vote_cor_reject.mean().item() * 100,
                maj_vote_cor_not_reject.shape[0],
                maj_vote_cor_not_reject.mean().item() * 100))

            # accuracy by rejection status within each class
            ttest_reject_by_class = []
            maj_vote_cor_reject_by_class = []
            maj_vote_cor_not_reject_by_class = []
            for c in range(config.data.num_classes):
                maj_vote_cor_class_c = majority_vote_correctness[true_y_label == c]  # (n_class_c, 1)
                ttest_reject_class_c = ttest_reject[true_y_label == c]  # (n_class_c, 1)
                maj_vote_cor_rej_class_c = maj_vote_cor_class_c[ttest_reject_class_c]  # (n_rej_class_c, 1)
                maj_vote_cor_not_rej_class_c = maj_vote_cor_class_c[~ttest_reject_class_c]  # (n_not_rej_class_c, 1)
                logging.info(("\n\tClass {} ({} total instances, {:.4f}% rejection rate):" +
                              "\n\t\t{} rejected t-tests, mean accuracy {:.4f}%" +
                              "\n\t\t{} not-rejected t-tests, mean accuracy {:.4f}%").format(
                    c, ttest_reject_class_c.shape[0], np.mean(ttest_reject_class_c) * 100,
                    maj_vote_cor_rej_class_c.shape[0], maj_vote_cor_rej_class_c.mean().item() * 100,
                    maj_vote_cor_not_rej_class_c.shape[0], maj_vote_cor_not_rej_class_c.mean().item() * 100))
                ttest_reject_by_class.append(np.mean(ttest_reject_class_c))
                maj_vote_cor_reject_by_class.append(maj_vote_cor_rej_class_c.mean().item())
                maj_vote_cor_not_reject_by_class.append(maj_vote_cor_not_rej_class_c.mean().item())

            return maj_vote_acc_by_class, ttest_rej_cor_pred_by_class, ttest_rej_incor_pred_by_class, \
                ttest_reject_by_class, maj_vote_cor_reject_by_class, maj_vote_cor_not_reject_by_class

        #####################################################################################################
        #####################################################################################################

        args = self.args
        config = self.config
        split = args.split
        log_path = os.path.join(self.args.log_path)

        references = pd.read_csv(self.csv_file)

        train_bags = references["train"].apply(
            lambda x: os.path.join(self.config.data.feature_path, x + ".h5")).values.tolist()

        def func_val(x):
            value = None
            if isinstance(x, str):
                value = os.path.join(self.config.data.feature_path, x + ".h5")
            return value

        val_bags = references.apply(lambda row: func_val(row.val), axis=1).dropna().values.tolist()

        test_bags = references.apply(lambda row: func_val(row.test), axis=1).dropna().values.tolist()

        train_dataloader = Whole_Slide_Bag(train_bags, label_file=self.config.data.label_file, dataset='camelyon')
        val_dataloader = Whole_Slide_Bag(val_bags, label_file=self.config.data.label_file, dataset='camelyon')
        test_dataloader = Whole_Slide_Bag(test_bags, label_file=self.config.data.label_file, dataset='camelyon')

        train_dataloader = torch.utils.data.DataLoader(train_dataloader, batch_size=1, shuffle=True)
        val_dataloader = torch.utils.data.DataLoader(val_dataloader, batch_size=1, shuffle=True)
        test_dataloader = torch.utils.data.DataLoader(test_dataloader, batch_size=1, shuffle=False)

        # use test batch size for training set during inference


        model = ConditionalModel(config, guidance=config.diffusion.include_guidance)

        if getattr(self.config.testing, "ckpt_id", None) is None:
            if args.eval_best:
                ckpt_id = 'best'
                states = torch.load(os.path.join(log_path, f"ckpt_{ckpt_id}.pth"),
                                    map_location=self.device)
            else:
                ckpt_id = 'last'
                states = torch.load(os.path.join(log_path, "ckpt.pth"),
                                    map_location=self.device)
        else:
            states = torch.load(os.path.join(log_path, f"ckpt_{self.config.testing.ckpt_id}.pth"),
                                map_location=self.device)
            ckpt_id = self.config.testing.ckpt_id

        logging.info(f"Loading from: {log_path}/ckpt_{ckpt_id}.pth")
        model = model.to(self.device)
        model.load_state_dict(states[0], strict=True)

        if self.config.model.ema:
            ema_helper = EMA(mu=self.config.model.ema_rate)
            ema_helper.register(model)
            ema_helper.load_state_dict(states[-1])
            ema_helper.ema(model)
        else:
            ema_helper = None

        model.eval()
        if args.sanity_check:
            logging.info("Evaluation function implementation sanity check...")
            config.testing.n_samples = 10
        if args.test_sample_seed >= 0:
            logging.info(f"Manually setting seed {args.test_sample_seed} for test time sampling of class prototype...")
            set_random_seed(args.test_sample_seed)

        # load auxiliary model
        if config.diffusion.apply_aux_cls:
            if hasattr(config.diffusion, "trained_aux_cls_ckpt_path"):
                aux_states = torch.load(os.path.join(config.diffusion.trained_aux_cls_ckpt_path,
                                                     config.diffusion.trained_aux_cls_ckpt_name),
                                        map_location=self.device)
                self.cond_pred_model.load_state_dict(aux_states['state_dict'], strict=True)
            else:
                aux_cls_path = log_path
                if hasattr(config.diffusion, "trained_aux_cls_log_path"):
                    aux_cls_path = config.diffusion.trained_aux_cls_log_path
                aux_states = torch.load(os.path.join(aux_cls_path, "aux_ckpt.pth"),
                                        map_location=self.device)
                self.cond_pred_model.load_state_dict(aux_states[0], strict=True)
            self.cond_pred_model.eval()
        # report test set RMSE if applied joint training
        y_acc_aux_model = self.evaluate_guidance_model(test_dataloader)
        logging.info("After training, guidance classifier accuracy on the test set is {:.8f}.".format(
            y_acc_aux_model))

        if config.testing.compute_metric_all_steps:
            logging.info("\nWe compute classification task metrics for all steps.\n")
        else:
            logging.info("\nWe pick samples at timestep t={} to compute evaluation metrics.\n".format(
                config.testing.metrics_t))

        # tune the scaling temperature parameter with training set
        T_description = "default"
        if args.tune_T:
            y_0_one_sample_all = []
            n_tune_T_samples = 5  # config.testing.n_samples; 25; 10
            logging.info("Begin generating {} samples for tuning temperature scaling parameter...".format(
                n_tune_T_samples))

            for idx,  feature_label_set in enumerate(train_dataloader):

                train_features, train_indices, train_values, y_labels_batch = feature_label_set
                train_features = train_features.to(self.device)
                train_indices = train_indices.to(self.device)
                train_values = train_values.to(self.device)

                # compute y_0_hat as the initial prediction to guide the reverse diffusion process

                y_0_hat_batch = self.compute_guiding_prediction([
                    torch.squeeze(train_features).to(self.device),
                    torch.squeeze(train_indices).to(self.device),
                    torch.squeeze(train_values).to(self.device)
                ])

                batch_size = train_features.shape[0]

                # x_batch with shape (batch_size, flattened_image_dim)
                x_tile = (train_features.repeat(n_tune_T_samples, 1, 1).transpose(0, 1)).to(
                        self.device).flatten(0, 1)

                y_0_hat_tile = (y_0_hat_batch.repeat(n_tune_T_samples, 1, 1).transpose(0, 1)).flatten(0, 1)
                y_T_mean_tile = y_0_hat_tile

                if config.diffusion.noise_prior:  # apply 0 instead of f_phi(x) as prior mean
                    y_T_mean_tile = torch.zeros(y_0_hat_tile.shape).to(self.device)

                minibatch_sample_start = time.time()
                y_0_sample_batch = p_sample_loop_only_y_0(model, x_tile, y_0_hat_tile, y_T_mean_tile,
                                                          self.num_timesteps,
                                                          self.alphas, self.one_minus_alphas_bar_sqrt)  # .cpu().numpy()

                # take the mean of n_tune_T_samples reconstructed y prototypes as the raw predicted y for T tuning

                y_0_sample_batch = y_0_sample_batch.reshape(batch_size,
                                                            n_tune_T_samples,
                                                            config.data.num_classes).mean(1)  # (batch_size, n_classes)
                minibatch_sample_end = time.time()
                logging.info("Minibatch {} sampling took {:.4f} seconds.".format(
                    idx, (minibatch_sample_end - minibatch_sample_start)))
                y_0_one_sample_all.append(y_0_sample_batch.detach())
                # only generate a few batches for sanity checking
                if args.sanity_check:
                    if idx > 2:
                        break


            logging.info("Begin optimizing temperature scaling parameter...")
            scale_T_raw = nn.Parameter(torch.zeros(1))
            scale_T_lr = 0.01
            scale_T_optimizer = torch.optim.Adam([scale_T_raw], lr=scale_T_lr)
            nll_losses = []
            scale_T_n_epochs = 10 if args.sanity_check else 1

            for _ in range(scale_T_n_epochs):
                for idx, feature_label_set in tqdm(enumerate(train_dataloader)):

                    train_features, train_indices, train_values, y_labels_batch = feature_label_set
                    y_one_hot_batch, y_logits_batch = cast_bag_label_to_one_hot_and_prototype(y_labels_batch,
                                                                                              config)
                    y_one_hot_batch = y_one_hot_batch.to(self.device)

                    scale_T = nn.functional.softplus(scale_T_raw).to(self.device)

                    y_0_sample_batch = y_0_one_sample_all[idx]

                    raw_p_val = compute_val_before_softmax(y_0_sample_batch)

                    # Eq. (9) of the Calibration paper (Guo et al. 2017)
                    y_0_scaled_batch = (raw_p_val / scale_T).softmax(1)

                    y_0_prob_batch = (y_0_scaled_batch * y_one_hot_batch).sum(1)  # instance likelihood
                    nll_loss = -torch.log(y_0_prob_batch).mean()
                    nll_losses.append(nll_loss.cpu().item())
                    # optimize scaling temperature T parameter
                    scale_T_optimizer.zero_grad()
                    nll_loss.backward()
                    scale_T_optimizer.step()
                    # only tune a few batches for sanity checking
                    if args.sanity_check:
                        if idx > 2:
                            break
                logging.info("NLL of the last mini-batch: {:.8f}".format(nll_losses[-1]))
            self.tuned_scale_T = nn.functional.softplus(scale_T_raw).detach().item()
            T_description = "tuned"
        else:
            self.tuned_scale_T = 1
        logging.info("Apply {} temperature scaling parameter T with a value of {:.4f}".format(
            T_description, self.tuned_scale_T))

        with torch.no_grad():
            true_y_label_by_batch_list = []
            majority_vote_by_batch_list = [[] for _ in range(self.num_timesteps + 1)]
            all_class_probs_by_batch_list = [[] for _ in range(self.num_timesteps + 1)]
            label_probs_by_batch_list = [[] for _ in range(self.num_timesteps + 1)]
            label_mean_probs_by_batch_list = [[] for _ in range(self.num_timesteps + 1)]
            instance_accuracy_by_batch_list = [[] for _ in range(self.num_timesteps + 1)]
            CI_by_batch_list = [[] for _ in range(self.num_timesteps + 1)]
            ttest_reject_by_batch_list = [[] for _ in range(self.num_timesteps + 1)]

            for step, feature_label_set in tqdm(enumerate(test_dataloader)):

                test_features, test_indices, test_values, y_labels_batch = feature_label_set
                test_features = test_features.to(self.device)
                test_indices = test_indices.to(self.device)
                test_values = test_values.to(self.device)

                # compute y_0_hat as the initial prediction to guide the reverse diffusion process

                y_0_hat_batch = self.compute_guiding_prediction([
                    torch.squeeze(test_features).to(self.device),
                    torch.squeeze(test_indices).to(self.device),
                    torch.squeeze(test_values).to(self.device)
                ])

                true_y_label_by_batch_list.append(y_labels_batch.numpy())
                # # record unflattened x as input to guidance aux classifier
                # x_unflat_batch = x_batch.to(self.device)

                # y_one_hot_batch, y_logits_batch = cast_label_to_one_hot_and_prototype(y_labels_batch, config)
                y_labels_batch = y_labels_batch.reshape(-1, 1)
                batch_size = test_features.shape[0]
                x_tile = (test_features.repeat(config.testing.n_samples, 1, 1).transpose(0, 1)).to(
                        self.device).flatten(0, 1)

                y_0_hat_tile = (y_0_hat_batch.repeat(config.testing.n_samples, 1, 1).transpose(0, 1)).flatten(0, 1)
                y_T_mean_tile = y_0_hat_tile
                if config.diffusion.noise_prior:  # apply 0 instead of f_phi(x) as prior mean
                    y_T_mean_tile = torch.zeros(y_0_hat_tile.shape).to(self.device)
                # generate samples from all time steps for the current mini-batch
                p_sample_loop_with_eval(model, x_tile, y_0_hat_tile, y_T_mean_tile, self.num_timesteps,
                                        self.alphas, self.one_minus_alphas_bar_sqrt,
                                        batch_size, config)
                # only compute on a few batches for sanity checking
                if args.sanity_check:
                    if step > 0:  # first two mini-batches
                        break


        all_true_y_labels = np.concatenate(true_y_label_by_batch_list, axis=0).reshape(-1, 1)
        y_majority_vote_accuracy_all_steps_list = []

        if config.testing.compute_metric_all_steps:
            for idx in range(self.num_timesteps + 1):
                current_t = self.num_timesteps - idx
                # compute accuracy at time step t
                majority_voted_class_t = majority_vote_by_batch_list[current_t]
                majority_vote_correctness = (majority_voted_class_t == all_true_y_labels)  # (n_test, 1)
                y_majority_vote_accuracy = np.mean(majority_vote_correctness)
                y_majority_vote_accuracy_all_steps_list.append(y_majority_vote_accuracy)
            logging.info(
                f"Majority Vote Accuracy across all steps: {y_majority_vote_accuracy_all_steps_list}.\n")
        else:
            # compute accuracy at time step metrics_t
            majority_voted_class_t = majority_vote_by_batch_list[config.testing.metrics_t]
            majority_vote_correctness = (majority_voted_class_t == all_true_y_labels)  # (n_test, 1)
            y_majority_vote_accuracy = np.mean(majority_vote_correctness)
            y_majority_vote_accuracy_all_steps_list.append(y_majority_vote_accuracy)
        instance_accuracy_t = instance_accuracy_by_batch_list[config.testing.metrics_t]
        logging.info("Mean accuracy of all samples at test instance level is {:.4f}%.\n".format(
            np.mean(instance_accuracy_t) * 100))

        # clear the memory
        plt.close('all')
        del label_probs_by_batch_list
        del majority_vote_by_batch_list
        del instance_accuracy_by_batch_list
        gc.collect()

        return []






