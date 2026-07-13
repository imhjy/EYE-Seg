from lr_schedular.scheduler import unet_scheduler, step, multistep, exponential, cosine_anneal_warm

__all__ = {
    'unet_scheduler': unet_scheduler,
    'step': step,
    'multistep': multistep,
    'exponential': exponential,
    'cosine_anneal_warm': cosine_anneal_warm
}


def build_lr_schedular(optimizer, hypes, **kwargs):
    lr_schedular_name = hypes['lr_scheduler']['core_method']

    return __all__[lr_schedular_name](optimizer, **kwargs)
