import argparse
import os
import time
import torch
import torch.autograd as autograd
import torch.distributed as dist
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, DistributedSampler
from data import PairedBratsValDataset, UnpairedBratsDataset, build_transform
from afd_utils import AdaptiveFrequencyDecomposition, adaptive_frequency_consistency_loss
from models_syndiff import apply_ema_to_generators, build_ema_helpers, build_syndiff_core_models, ddp_wrap_models, restore_generators_from_ema, set_requires_grad, update_ema_helpers, unwrap_model
from train_syndiff import Diffusion_Coefficients, Posterior_Coefficients, all_finite, amp_ctx, cleanup_ddp, load_full_checkpoint, q_sample_pairs, sample_posterior, save_A_to_B_weights, save_case_metrics_to_xlsx, save_full_checkpoint, setup_ddp, softplus_fake, softplus_real, validate_A_to_B_case_level, zero_optimizers
from utils_ import append_csv, is_main_process, seed_everything, seed_worker

def str2bool(v):
    if isinstance(v, bool):
        return v
    v = str(v).lower()
    if v in ('yes', 'true', 't', '1', 'y'):
        return True
    if v in ('no', 'false', 'f', '0', 'n'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    parser = argparse.ArgumentParser('SynDiff with FDDT')
    parser.add_argument('--dataroot', type=str, default='')
    parser.add_argument('--checkpoints_dir', type=str, default='')
    parser.add_argument('--name', type=str, default='fddt_syndiff')
    parser.add_argument('--source_domain', type=str, default='')
    parser.add_argument('--target_domain', type=str, default='')
    parser.add_argument('--eval_phase', type=str, default='val', choices=['val', 'test'])
    parser.add_argument('--eval_pairing', type=str, default='key', choices=['key', 'order'])
    parser.add_argument('--case_id_regex', type=str, default='')
    parser.add_argument('--input_nc', type=int, default=1)
    parser.add_argument('--output_nc', type=int, default=1)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--load_size', type=int, default=256)
    parser.add_argument('--crop_size', type=int, default=0)
    parser.add_argument('--no_flip', action='store_true', default=True)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--prefetch_factor', type=int, default=4)
    parser.add_argument('--diff_base_ch', type=int, default=48)
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--resnet_blocks', type=int, default=6)
    parser.add_argument('--spectral_norm', type=str2bool, default=True)
    parser.add_argument('--nz', type=int, default=100)
    parser.add_argument('--t_emb_dim', type=int, default=256)
    parser.add_argument('--num_timesteps', type=int, default=5)
    parser.add_argument('--beta_min', type=float, default=0.1)
    parser.add_argument('--beta_max', type=float, default=20.0)
    parser.add_argument('--use_geometric', action='store_true', default=False)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--val_batch_size', type=int, default=0)
    parser.add_argument('--num_epoch', type=int, default=500)
    parser.add_argument('--lr_g', type=float, default=1e-05)
    parser.add_argument('--lr_d', type=float, default=1e-05)
    parser.add_argument('--weight_decay', type=float, default=None)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.9)
    parser.add_argument('--no_lr_decay', action='store_true', default=False)
    parser.add_argument('--lambda_l1_loss', type=float, default=10.0)
    parser.add_argument('--r1_gamma', type=float, default=0.05)
    parser.add_argument('--lazy_reg', type=int, default=10)
    parser.add_argument('--grad_clip', type=float, default=0.0)
    parser.add_argument('--amp', type=str2bool, default=True)
    parser.add_argument('--safe_skip_nan', type=str2bool, default=True)
    parser.add_argument('--max_nan_batches', type=int, default=20)
    parser.add_argument('--use_ema', type=str2bool, default=True)
    parser.add_argument('--ema_decay', type=float, default=0.999)
    parser.add_argument('--eval_with_ema', type=str2bool, default=True)
    parser.add_argument('--seed', type=int, default=1024)
    parser.add_argument('--val_seed', type=int, default=1234)
    parser.add_argument('--print_freq', type=int, default=500)
    parser.add_argument('--save_content_every', type=int, default=10)
    parser.add_argument('--save_epoch_weights', action='store_true', default=False)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--min_delta_ssim', type=float, default=1e-05)
    parser.add_argument('--min_delta_psnr', type=float, default=0.001)
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--lambda_freq', type=float, default=None)
    parser.add_argument('--afd_embed_dim', type=int, default=None)
    parser.add_argument('--afd_num_heads', type=int, default=None)
    parser.add_argument('--afd_high_min', type=float, default=None)
    parser.add_argument('--afd_high_max', type=float, default=None)
    parser.add_argument('--afd_low_min', type=float, default=None)
    parser.add_argument('--afd_low_max', type=float, default=None)
    parser.add_argument('--afd_input_size', type=int, default=None)
    parser.add_argument('--afd_resume', type=str, default='')
    parser.add_argument('--freq_bidir', type=str2bool, default=False)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--local-rank', type=int, default=-1)
    args = parser.parse_args()
    required = {'dataroot': args.dataroot, 'checkpoints_dir': args.checkpoints_dir, 'source_domain': args.source_domain, 'target_domain': args.target_domain, 'case_id_regex': args.case_id_regex, 'weight_decay': args.weight_decay, 'lambda_freq': args.lambda_freq, 'afd_embed_dim': args.afd_embed_dim, 'afd_num_heads': args.afd_num_heads, 'afd_high_min': args.afd_high_min, 'afd_high_max': args.afd_high_max, 'afd_low_min': args.afd_low_min, 'afd_low_max': args.afd_low_max, 'afd_input_size': args.afd_input_size}
    missing = [key for key, value in required.items() if value in (None, '')]
    if missing:
        parser.error('Required settings are empty: ' + ', '.join(missing))
    return args

def make_zero(device):
    return torch.zeros((), device=device, dtype=torch.float32)

def save_full_checkpoint_with_afd(path, epoch, global_step, models, optimizers, schedulers, scaler, args, best, ema_helpers, afd):
    save_full_checkpoint(path, epoch, global_step, models, optimizers, schedulers, scaler, args, best, ema_helpers)
    checkpoint = torch.load(path, map_location='cpu')
    checkpoint['afd'] = unwrap_model(afd).state_dict()
    torch.save(checkpoint, path)

def main():
    args = parse_args()
    rank, world_size, local_rank = setup_ddp()
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    if world_size > 1 and args.safe_skip_nan:
        if is_main_process():
            print('[DDP] safe_skip_nan is disabled under DDP to avoid reducer-state mismatch.')
        args.safe_skip_nan = False
    seed_everything(args.seed + rank)
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision('high')
    except Exception:
        pass
    exp_dir = os.path.join(args.checkpoints_dir, args.name)
    xlsx_path = os.path.join(exp_dir, 'case_level_metrics.xlsx')
    metrics_csv = os.path.join(exp_dir, 'metrics.csv')
    if is_main_process():
        os.makedirs(exp_dir, exist_ok=True)
        print(f'[DDP] world_size={world_size}')
        print(f'[AMP] enabled={args.amp}, GradScaler=True, safe_skip_nan={args.safe_skip_nan}')
        print(f'[Data] train source: {args.dataroot}/train/{args.source_domain}')
        print(f'[Data] eval source: {args.dataroot}/{args.eval_phase}/{args.source_domain}')
        print(f'[SynDiff-core] T={args.num_timesteps}, beta_min={args.beta_min}, beta_max={args.beta_max}, geometric={args.use_geometric}')
        print(f'[Model] diff_base_ch={args.diff_base_ch}, ngf={args.ngf}, ndf={args.ndf}, spectral_norm={args.spectral_norm}')
        print(f'[FDDT] lambda_freq={args.lambda_freq}, adaptive_cutoffs=True, bidir={args.freq_bidir}')
        print(f'[Save] {exp_dir}')
    train_tf = build_transform('train', args.input_nc, args.load_size, args.crop_size, no_flip=True)
    val_tf = build_transform('val', args.input_nc, args.load_size, args.crop_size, no_flip=True)
    train_set = UnpairedBratsDataset(args.dataroot, 'train', train_tf, source_domain=args.source_domain, target_domain=args.target_domain)
    val_set = PairedBratsValDataset(args.dataroot, args.eval_phase, val_tf, source_domain=args.source_domain, target_domain=args.target_domain, pairing=args.eval_pairing)
    train_sampler = DistributedSampler(train_set, shuffle=True, drop_last=True) if world_size > 1 else None
    val_sampler = DistributedSampler(val_set, shuffle=False, drop_last=False) if world_size > 1 else None
    loader_kwargs = dict(num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0, worker_init_fn=seed_worker)
    if args.num_workers > 0:
        loader_kwargs['prefetch_factor'] = args.prefetch_factor
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=train_sampler is None, sampler=train_sampler, drop_last=True, **loader_kwargs)
    val_bs = args.val_batch_size if args.val_batch_size > 0 else args.batch_size * 4
    val_loader = DataLoader(val_set, batch_size=val_bs, shuffle=False, sampler=val_sampler, drop_last=False, **loader_kwargs)
    coeff = Diffusion_Coefficients(args, device)
    pos_coeff = Posterior_Coefficients(args, device)
    models = build_syndiff_core_models(args, device)
    afd = AdaptiveFrequencyDecomposition(input_nc=args.input_nc, embed_dim=args.afd_embed_dim, num_heads=args.afd_num_heads, high_min=args.afd_high_min, high_max=args.afd_high_max, low_min=args.afd_low_min, low_max=args.afd_low_max).to(device)
    with torch.no_grad():
        afd(torch.zeros(1, args.input_nc, args.afd_input_size, args.afd_input_size, device=device))
    if args.afd_resume:
        afd.load_state_dict(torch.load(args.afd_resume, map_location=device), strict=True)
    optimizers = {'gen_diffusive_1': torch.optim.AdamW(models.gen_diffusive_1.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay), 'gen_diffusive_2': torch.optim.AdamW(models.gen_diffusive_2.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay), 'gen_non_diffusive_1to2': torch.optim.AdamW(models.gen_non_diffusive_1to2.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay), 'gen_non_diffusive_2to1': torch.optim.AdamW(models.gen_non_diffusive_2to1.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay), 'disc_diffusive_1': torch.optim.AdamW(models.disc_diffusive_1.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay), 'disc_diffusive_2': torch.optim.AdamW(models.disc_diffusive_2.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay), 'disc_cycle_1': torch.optim.AdamW(models.disc_cycle_1.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay), 'disc_cycle_2': torch.optim.AdamW(models.disc_cycle_2.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay), 'afd': torch.optim.AdamW(afd.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)}
    schedulers = {}
    if not args.no_lr_decay:
        for k, opt in optimizers.items():
            schedulers[k] = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.num_epoch, eta_min=1e-05)
    scaler = GradScaler(enabled=args.amp and device.type == 'cuda')
    ema_helpers = build_ema_helpers(args, models)
    start_epoch = 0
    global_step = 0
    best = {'best_ssim': -1000000000.0, 'best_psnr': -1000000000.0, 'best_ssim_epoch': -1, 'best_psnr_epoch': -1, 'bad_epochs': 0}
    if args.resume:
        ckpt = load_full_checkpoint(args.resume, models, optimizers, schedulers, scaler, ema_helpers, device)
        start_epoch = int(ckpt['epoch']) + 1
        global_step = int(ckpt.get('global_step', 0))
        best.update(ckpt.get('best', {}))
        if 'afd' in ckpt:
            afd.load_state_dict(ckpt['afd'], strict=True)
        elif not args.afd_resume:
            raise RuntimeError('The resume checkpoint does not contain AFD weights. Supply --afd_resume explicitly.')
        if is_main_process():
            print(f'[Resume] loaded {args.resume}, start_epoch={start_epoch}')
    models = ddp_wrap_models(models, local_rank=local_rank, find_unused_parameters=False)
    if world_size > 1:
        afd = torch.nn.parallel.DistributedDataParallel(afd, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    nan_batches = 0
    try:
        for epoch in range(start_epoch, args.num_epoch + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            epoch_start = time.time()
            for net in [models.gen_diffusive_1, models.gen_diffusive_2, models.gen_non_diffusive_1to2, models.gen_non_diffusive_2to1, models.disc_diffusive_1, models.disc_diffusive_2, models.disc_cycle_1, models.disc_cycle_2]:
                net.train()
            afd.train()
            loss_acc = {'G_sum': 0.0, 'G_cycle': 0.0, 'G_L1': 0.0, 'G_adv': 0.0, 'G_cycle_adv': 0.0, 'G_freq': 0.0, 'G_fddt': 0.0, 'G_fddt_A2B': 0.0, 'G_fddt_B2A': 0.0, 'G_fddt_low': 0.0, 'G_fddt_high': 0.0, 'D_diff': 0.0, 'D_cycle': 0.0, 'R1': 0.0}
            n_iter = 0
            for batch in train_loader:
                real_data1 = batch['A'].to(device, non_blocking=True)
                real_data2 = batch['B'].to(device, non_blocking=True)
                bsz = real_data1.size(0)
                r1_value = make_zero(device)
                set_requires_grad([models.disc_diffusive_1, models.disc_diffusive_2, models.disc_cycle_1, models.disc_cycle_2], True)
                for k in ['disc_diffusive_1', 'disc_diffusive_2', 'disc_cycle_1', 'disc_cycle_2']:
                    optimizers[k].zero_grad(set_to_none=True)
                t1 = torch.randint(0, args.num_timesteps, (bsz,), device=device)
                t2 = torch.randint(0, args.num_timesteps, (bsz,), device=device)
                x1_t, x1_tp1 = q_sample_pairs(coeff, real_data1, t1)
                x2_t, x2_tp1 = q_sample_pairs(coeff, real_data2, t2)
                with torch.no_grad():
                    z1 = torch.randn(bsz, args.nz, device=device)
                    z2 = torch.randn(bsz, args.nz, device=device)
                    G_2to1 = unwrap_model(models.gen_non_diffusive_2to1)
                    G_1to2 = unwrap_model(models.gen_non_diffusive_1to2)
                    G_diff_1 = unwrap_model(models.gen_diffusive_1)
                    G_diff_2 = unwrap_model(models.gen_diffusive_2)
                    x1_0_predict = G_2to1(real_data2)
                    x2_0_predict = G_1to2(real_data1)
                    with amp_ctx(args.amp):
                        x1_0_predict_diff = G_diff_1(torch.cat((x1_tp1.detach(), x2_0_predict.detach()), dim=1), t1, z1)
                        x2_0_predict_diff = G_diff_2(torch.cat((x2_tp1.detach(), x1_0_predict.detach()), dim=1), t2, z2)
                    x1_pos_sample = sample_posterior(pos_coeff, x1_0_predict_diff[:, [0], ...], x1_tp1, t1)
                    x2_pos_sample = sample_posterior(pos_coeff, x2_0_predict_diff[:, [0], ...], x2_tp1, t2)
                need_r1 = args.r1_gamma > 0 and (args.lazy_reg is None or global_step % args.lazy_reg == 0)
                x1_t_for_d = x1_t.detach().float()
                x2_t_for_d = x2_t.detach().float()
                if need_r1:
                    x1_t_for_d.requires_grad_(True)
                    x2_t_for_d.requires_grad_(True)
                with amp_ctx(args.amp and (not need_r1)):
                    D1_real = models.disc_diffusive_1(x1_t_for_d, t1, x1_tp1.detach().float()).view(-1)
                    D2_real = models.disc_diffusive_2(x2_t_for_d, t2, x2_tp1.detach().float()).view(-1)
                with amp_ctx(args.amp):
                    D1_fake = models.disc_diffusive_1(x1_pos_sample.detach(), t1, x1_tp1.detach()).view(-1)
                    D2_fake = models.disc_diffusive_2(x2_pos_sample.detach(), t2, x2_tp1.detach()).view(-1)
                errD_diff = softplus_real(D1_real) + softplus_real(D2_real) + softplus_fake(D1_fake) + softplus_fake(D2_fake)
                ok, bad_name = all_finite({'D_diff': errD_diff})
                if not ok:
                    nan_batches += 1
                    zero_optimizers(optimizers)
                    if is_main_process():
                        print(f'[SafeSkip] non-finite {bad_name} at step={global_step}, skipped={nan_batches}')
                    if not args.safe_skip_nan or nan_batches > args.max_nan_batches:
                        raise RuntimeError(f'Non-finite {bad_name}')
                    continue
                scaler.scale(errD_diff).backward(retain_graph=need_r1)
                if need_r1:
                    grad1 = autograd.grad(outputs=D1_real.sum(), inputs=x1_t_for_d, create_graph=True, retain_graph=True, only_inputs=True)[0]
                    grad2 = autograd.grad(outputs=D2_real.sum(), inputs=x2_t_for_d, create_graph=True, retain_graph=True, only_inputs=True)[0]
                    r1_loss = args.r1_gamma * 0.5 * (grad1.flatten(1).pow(2).sum(1).mean() + grad2.flatten(1).pow(2).sum(1).mean())
                    ok, bad_name = all_finite({'R1': r1_loss})
                    if not ok:
                        nan_batches += 1
                        zero_optimizers(optimizers)
                        if is_main_process():
                            print(f'[SafeSkip] non-finite {bad_name} at step={global_step}, skipped={nan_batches}')
                        if not args.safe_skip_nan or nan_batches > args.max_nan_batches:
                            raise RuntimeError(f'Non-finite {bad_name}')
                        continue
                    scaler.scale(r1_loss).backward()
                    r1_value = r1_loss.detach()
                with torch.no_grad():
                    G_2to1 = unwrap_model(models.gen_non_diffusive_2to1)
                    G_1to2 = unwrap_model(models.gen_non_diffusive_1to2)
                    x1_0_predict = G_2to1(real_data2)
                    x2_0_predict = G_1to2(real_data1)
                with amp_ctx(args.amp):
                    D_cycle1_real = models.disc_cycle_1(real_data1).view(-1)
                    D_cycle2_real = models.disc_cycle_2(real_data2).view(-1)
                    D_cycle1_fake = models.disc_cycle_1(x1_0_predict.detach()).view(-1)
                    D_cycle2_fake = models.disc_cycle_2(x2_0_predict.detach()).view(-1)
                    errD_cycle = softplus_real(D_cycle1_real) + softplus_real(D_cycle2_real) + softplus_fake(D_cycle1_fake) + softplus_fake(D_cycle2_fake)
                ok, bad_name = all_finite({'D_cycle': errD_cycle})
                if not ok:
                    nan_batches += 1
                    zero_optimizers(optimizers)
                    if is_main_process():
                        print(f'[SafeSkip] non-finite {bad_name} at step={global_step}, skipped={nan_batches}')
                    if not args.safe_skip_nan or nan_batches > args.max_nan_batches:
                        raise RuntimeError(f'Non-finite {bad_name}')
                    continue
                scaler.scale(errD_cycle).backward()
                for k in ['disc_diffusive_1', 'disc_diffusive_2', 'disc_cycle_1', 'disc_cycle_2']:
                    scaler.unscale_(optimizers[k])
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(models.disc_diffusive_1.parameters(), args.grad_clip)
                    torch.nn.utils.clip_grad_norm_(models.disc_diffusive_2.parameters(), args.grad_clip)
                    torch.nn.utils.clip_grad_norm_(models.disc_cycle_1.parameters(), args.grad_clip)
                    torch.nn.utils.clip_grad_norm_(models.disc_cycle_2.parameters(), args.grad_clip)
                for k in ['disc_diffusive_1', 'disc_diffusive_2', 'disc_cycle_1', 'disc_cycle_2']:
                    scaler.step(optimizers[k])
                set_requires_grad([models.disc_diffusive_1, models.disc_diffusive_2, models.disc_cycle_1, models.disc_cycle_2], False)
                for k in ['gen_diffusive_1', 'gen_diffusive_2', 'gen_non_diffusive_1to2', 'gen_non_diffusive_2to1', 'afd']:
                    optimizers[k].zero_grad(set_to_none=True)
                t1 = torch.randint(0, args.num_timesteps, (bsz,), device=device)
                t2 = torch.randint(0, args.num_timesteps, (bsz,), device=device)
                x1_t, x1_tp1 = q_sample_pairs(coeff, real_data1, t1)
                x2_t, x2_tp1 = q_sample_pairs(coeff, real_data2, t2)
                z1 = torch.randn(bsz, args.nz, device=device)
                z2 = torch.randn(bsz, args.nz, device=device)
                D_diff_1 = unwrap_model(models.disc_diffusive_1)
                D_diff_2 = unwrap_model(models.disc_diffusive_2)
                D_cyc_1 = unwrap_model(models.disc_cycle_1)
                D_cyc_2 = unwrap_model(models.disc_cycle_2)
                with amp_ctx(args.amp):
                    x1_0_predict = models.gen_non_diffusive_2to1(real_data2)
                    x2_0_predict_cycle = models.gen_non_diffusive_1to2(x1_0_predict)
                    x2_0_predict = models.gen_non_diffusive_1to2(real_data1)
                    x1_0_predict_cycle = models.gen_non_diffusive_2to1(x2_0_predict)
                    x1_0_predict_diff = models.gen_diffusive_1(torch.cat((x1_tp1.detach(), x2_0_predict), dim=1), t1, z1)
                    x2_0_predict_diff = models.gen_diffusive_2(torch.cat((x2_tp1.detach(), x1_0_predict), dim=1), t2, z2)
                    x1_pos_sample = sample_posterior(pos_coeff, x1_0_predict_diff[:, [0], ...], x1_tp1, t1)
                    x2_pos_sample = sample_posterior(pos_coeff, x2_0_predict_diff[:, [0], ...], x2_tp1, t2)
                    output1 = D_diff_1(x1_pos_sample, t1, x1_tp1.detach()).view(-1)
                    output2 = D_diff_2(x2_pos_sample, t2, x2_tp1.detach()).view(-1)
                    errG_adv = softplus_real(output1) + softplus_real(output2)
                    D_cycle1_fake = D_cyc_1(x1_0_predict).view(-1)
                    D_cycle2_fake = D_cyc_2(x2_0_predict).view(-1)
                    errG_cycle_adv = softplus_real(D_cycle1_fake) + softplus_real(D_cycle2_fake)
                    errG_L1 = F.l1_loss(x1_0_predict_diff[:, [0], ...].float(), real_data1.float()) + F.l1_loss(x2_0_predict_diff[:, [0], ...].float(), real_data2.float())
                    errG_cycle = F.l1_loss(x1_0_predict_cycle.float(), real_data1.float()) + F.l1_loss(x2_0_predict_cycle.float(), real_data2.float())
                with amp_ctx(False):
                    errG_fddt_A2B_raw, errG_fddt_A2B_low, errG_fddt_A2B_high = adaptive_frequency_consistency_loss(generator=models.gen_non_diffusive_1to2, afd=afd, source=real_data1.float(), full_translation=x2_0_predict.float())[:3]
                    if args.freq_bidir:
                        errG_fddt_B2A_raw, errG_fddt_B2A_low, errG_fddt_B2A_high = adaptive_frequency_consistency_loss(generator=models.gen_non_diffusive_2to1, afd=afd, source=real_data2.float(), full_translation=x1_0_predict.float())[:3]
                    else:
                        errG_fddt_B2A_raw = make_zero(device)
                        errG_fddt_B2A_low = make_zero(device)
                        errG_fddt_B2A_high = make_zero(device)
                    errG_fddt_raw = errG_fddt_A2B_raw + errG_fddt_B2A_raw
                    errG_freq = float(args.lambda_freq) * errG_fddt_raw
                errG = args.lambda_l1_loss * errG_cycle + errG_adv + errG_cycle_adv + args.lambda_l1_loss * errG_L1 + errG_freq
                ok, bad_name = all_finite({'G_sum': errG, 'G_cycle': errG_cycle, 'G_L1': errG_L1, 'G_adv': errG_adv, 'G_cycle_adv': errG_cycle_adv, 'G_freq': errG_freq, 'G_fddt': errG_freq, 'G_fddt_A2B': errG_fddt_A2B_raw, 'G_fddt_B2A': errG_fddt_B2A_raw, 'G_fddt_low': errG_fddt_A2B_low + errG_fddt_B2A_low, 'G_fddt_high': errG_fddt_A2B_high + errG_fddt_B2A_high})
                if not ok:
                    nan_batches += 1
                    zero_optimizers(optimizers)
                    if is_main_process():
                        print(f'[SafeSkip] non-finite {bad_name} at step={global_step}, skipped={nan_batches}')
                    if not args.safe_skip_nan or nan_batches > args.max_nan_batches:
                        raise RuntimeError(f'Non-finite {bad_name}')
                    continue
                scaler.scale(errG).backward()
                for k in ['gen_diffusive_1', 'gen_diffusive_2', 'gen_non_diffusive_1to2', 'gen_non_diffusive_2to1', 'afd']:
                    scaler.unscale_(optimizers[k])
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(models.gen_diffusive_1.parameters(), args.grad_clip)
                    torch.nn.utils.clip_grad_norm_(models.gen_diffusive_2.parameters(), args.grad_clip)
                    torch.nn.utils.clip_grad_norm_(models.gen_non_diffusive_1to2.parameters(), args.grad_clip)
                    torch.nn.utils.clip_grad_norm_(models.gen_non_diffusive_2to1.parameters(), args.grad_clip)
                    torch.nn.utils.clip_grad_norm_(afd.parameters(), args.grad_clip)
                for k in ['gen_diffusive_1', 'gen_diffusive_2', 'gen_non_diffusive_1to2', 'gen_non_diffusive_2to1', 'afd']:
                    scaler.step(optimizers[k])
                scaler.update()
                update_ema_helpers(ema_helpers, models)
                nan_batches = 0
                global_step += 1
                n_iter += 1
                loss_acc['G_sum'] += float(errG.detach())
                loss_acc['G_cycle'] += float(errG_cycle.detach())
                loss_acc['G_L1'] += float(errG_L1.detach())
                loss_acc['G_adv'] += float(errG_adv.detach())
                loss_acc['G_cycle_adv'] += float(errG_cycle_adv.detach())
                loss_acc['G_freq'] += float(errG_freq.detach())
                loss_acc['G_fddt'] += float(errG_freq.detach())
                loss_acc['G_fddt_A2B'] += float(errG_fddt_A2B_raw.detach())
                loss_acc['G_fddt_B2A'] += float(errG_fddt_B2A_raw.detach())
                loss_acc['G_fddt_low'] += float((errG_fddt_A2B_low + errG_fddt_B2A_low).detach())
                loss_acc['G_fddt_high'] += float((errG_fddt_A2B_high + errG_fddt_B2A_high).detach())
                loss_acc['D_diff'] += float(errD_diff.detach())
                loss_acc['D_cycle'] += float(errD_cycle.detach())
                loss_acc['R1'] += float(r1_value.detach())
                if is_main_process() and global_step % args.print_freq == 0:
                    print(f'[Epoch {epoch:03d}/{args.num_epoch}] iter={n_iter:04d} G-Cycle={float(errG_cycle.detach()):.4f} G-L1={float(errG_L1.detach()):.4f} G-Adv={float(errG_adv.detach()):.4f} G-cycle-Adv={float(errG_cycle_adv.detach()):.4f} Freq={float(errG_freq.detach()):.4f} FDDT={float(errG_freq.detach()):.4f} G-Sum={float(errG.detach()):.4f} D={float(errD_diff.detach()):.4f} D-cycle={float(errD_cycle.detach()):.4f} R1={float(r1_value.detach()):.4f} scale={scaler.get_scale():.1f}')
            for sch in schedulers.values():
                sch.step()
            case_rows, summary = validate_A_to_B_case_level(models, val_loader, device, args, pos_coeff, ema_helpers)
            if is_main_process():
                save_case_metrics_to_xlsx(xlsx_path, epoch, case_rows, summary)
                avg = {k: v / max(n_iter, 1) for k, v in loss_acc.items()}
                lr_g = optimizers['gen_diffusive_2'].param_groups[0]['lr']
                lr_d = optimizers['disc_diffusive_2'].param_groups[0]['lr']
                cur_ssim = summary['case_ssim_mean']
                cur_psnr = summary['case_psnr_mean']
                improved = False
                if cur_ssim > best['best_ssim'] + args.min_delta_ssim:
                    best['best_ssim'] = cur_ssim
                    best['best_ssim_epoch'] = epoch
                    best['bad_epochs'] = 0
                    improved = True
                    best_ssim_path = os.path.join(exp_dir, 'best_ssim_netG_A2B.pth')
                    if args.eval_with_ema and ema_helpers:
                        apply_ema_to_generators(ema_helpers, models)
                        save_A_to_B_weights(best_ssim_path, models)
                        restore_generators_from_ema(ema_helpers, models)
                    else:
                        save_A_to_B_weights(best_ssim_path, models)
                    torch.save(unwrap_model(afd).state_dict(), os.path.join(exp_dir, 'best_ssim_AFD.pth'))
                if cur_psnr > best['best_psnr'] + args.min_delta_psnr:
                    best['best_psnr'] = cur_psnr
                    best['best_psnr_epoch'] = epoch
                    best['bad_epochs'] = 0
                    improved = True
                    best_psnr_path = os.path.join(exp_dir, 'best_psnr_netG_A2B.pth')
                    if args.eval_with_ema and ema_helpers:
                        apply_ema_to_generators(ema_helpers, models)
                        save_A_to_B_weights(best_psnr_path, models)
                        restore_generators_from_ema(ema_helpers, models)
                    else:
                        save_A_to_B_weights(best_psnr_path, models)
                    torch.save(unwrap_model(afd).state_dict(), os.path.join(exp_dir, 'best_psnr_AFD.pth'))
                if not improved:
                    best['bad_epochs'] += 1
                row = {'epoch': epoch, 'lr_g': lr_g, 'lr_d': lr_d, **avg, 'ssim_mean': cur_ssim, 'ssim_std': summary['case_ssim_std'], 'psnr_mean': cur_psnr, 'psnr_std': summary['case_psnr_std'], 'best_ssim': best['best_ssim'], 'best_psnr': best['best_psnr'], 'best_ssim_epoch': best['best_ssim_epoch'], 'best_psnr_epoch': best['best_psnr_epoch'], 'bad_epochs': best['bad_epochs'], 'time_min': (time.time() - epoch_start) / 60.0}
                append_csv(metrics_csv, row)
                save_full_checkpoint_with_afd(os.path.join(exp_dir, 'latest.pth'), epoch, global_step, models, optimizers, schedulers, scaler, args, best, ema_helpers, afd)
                torch.save(unwrap_model(afd).state_dict(), os.path.join(exp_dir, 'latest_AFD.pth'))
                if epoch % args.save_content_every == 0:
                    save_full_checkpoint_with_afd(os.path.join(exp_dir, f'content_epoch_{epoch:03d}.pth'), epoch, global_step, models, optimizers, schedulers, scaler, args, best, ema_helpers, afd)
                    torch.save(unwrap_model(afd).state_dict(), os.path.join(exp_dir, f'content_epoch_{epoch:03d}_AFD.pth'))
                if args.save_epoch_weights:
                    save_A_to_B_weights(os.path.join(exp_dir, f'netG_A2B_epoch_{epoch:03d}.pth'), models)
                    torch.save(unwrap_model(afd).state_dict(), os.path.join(exp_dir, f'netG_A2B_epoch_{epoch:03d}_AFD.pth'))
                print(f"[Epoch {epoch:03d}/{args.num_epoch}] lr_g={lr_g:.2e} lr_d={lr_d:.2e} G={avg['G_sum']:.4f} Gcyc={avg['G_cycle']:.4f} GL1={avg['G_L1']:.4f} Gadv={avg['G_adv']:.4f} GcycAdv={avg['G_cycle_adv']:.4f} Freq={avg['G_freq']:.4f} FDDT={avg['G_fddt']:.4f} D={avg['D_diff']:.4f} Dcyc={avg['D_cycle']:.4f} R1={avg['R1']:.4f} SSIM={cur_ssim:.5f}±{summary['case_ssim_std']:.5f} PSNR={cur_psnr:.3f}±{summary['case_psnr_std']:.3f} best_ssim={best['best_ssim']:.5f}@{best['best_ssim_epoch']} best_psnr={best['best_psnr']:.3f}@{best['best_psnr_epoch']} patience={best['bad_epochs']}/{args.patience} time={(time.time() - epoch_start) / 60.0:.1f}m")
            if is_main_process():
                stop_now = best['bad_epochs'] >= args.patience
            else:
                stop_now = False
            if dist.is_available() and dist.is_initialized():
                stop_flag = torch.tensor([1 if stop_now else 0], device=device, dtype=torch.int64)
                dist.broadcast(stop_flag, src=0)
                stop_now = bool(stop_flag.item())
            if stop_now:
                if is_main_process():
                    print(f"[EarlyStop] patience reached: {best['bad_epochs']}/{args.patience}")
                break
    finally:
        cleanup_ddp()
if __name__ == '__main__':
    main()
