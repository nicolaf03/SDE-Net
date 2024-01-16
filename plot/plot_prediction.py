from matplotlib import pyplot as plt

def plot_hist(real_samples, generated_samples, plot_locs, zone):
    for prop in plot_locs:
        time = int(prop * (real_samples.size(1) - 1))
        real_samples_time = real_samples[:, time]
        generated_samples_time = generated_samples[:, time]
        
        plt.figure()
        _, bins, _ = plt.hist(real_samples_time.cpu().numpy(), bins=32, alpha=0.7, label='Real', color='dodgerblue',
                              density=True)
        bin_width = bins[1] - bins[0]
        num_bins = int((generated_samples_time.max() - generated_samples_time.min()).item() // bin_width)
        plt.hist(generated_samples_time.cpu().numpy(), bins=num_bins, alpha=0.7, label='Generated', color='crimson',
                 density=True)
        plt.legend()
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.title(f'Marginal distribution at time {time}.')
        plt.tight_layout()
        #plt.savefig(f'./torchsde/images/{zone}/marginal_distribution_{prop}_{zone}.png', dpi=200, format='png')
        #plt.show()


def plot_samples(ts, real_samples, generated_samples, num_plot_samples, zone):
    real_samples = real_samples[:num_plot_samples]
    generated_samples = generated_samples[:num_plot_samples]
    
    plt.figure()
    real_first = True
    generated_first = True
    for real_sample_ in real_samples:
        kwargs = {'label': 'Real'} if real_first else {}
        plt.plot(ts.cpu(), real_sample_.cpu(), color='dodgerblue', linewidth=0.5, alpha=0.7, **kwargs)
        real_first = False
    for generated_sample_ in generated_samples:
        kwargs = {'label': 'Generated'} if generated_first else {}
        plt.plot(ts.cpu(), generated_sample_.cpu(), color='crimson', linewidth=0.5, alpha=0.7, **kwargs)
        generated_first = False
    plt.legend()
    plt.title(f"{num_plot_samples} samples from both real and generated distributions ({zone}).")
    plt.tight_layout()
    plt.savefig(f'images/samples_real_vs_generated_{zone}.png', dpi=200, format='png')
    #plt.show() 