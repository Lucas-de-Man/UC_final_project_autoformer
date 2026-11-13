from layers.AutoCorrelation import AutoCorrelation
import torch

class CombinedAutoCorrelation(AutoCorrelation):
    """
        AutoCorrelation where the correlation for any shift (tau) between keys and queries is multiplied by the
        respective Spectral Density value for that shift as follows:

        C(tau) = R(tau) * S(1/tau)

        where R(tau) is the autocorrelation for the shift tau, and S(f) is the spectral density for the hypothetical
        frequency matching the shift tau.
    """
    def __init__(self, mask_flag=True, factor=1, scale=None, attention_dropout=0.1, output_attention=False):
        super(CombinedAutoCorrelation, self).__init__(mask_flag, factor, scale, attention_dropout, output_attention)

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        if L > S:
            zeros = torch.zeros_like(queries[:, :(L - S), :]).float()
            values = torch.cat([values, zeros], dim=1)
            keys = torch.cat([keys, zeros], dim=1)
        else:
            values = values[:, :L, :, :]
            keys = keys[:, :L, :, :]

        # period-based dependencies
        q_fft = torch.fft.rfft(queries.permute(0, 2, 3, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(keys.permute(0, 2, 3, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, n=L, dim=-1)

        # make sure we use the same device as the data it interacts with
        periods = torch.arange(0, L, device=corr.get_device())
        period_map = torch.empty(L, dtype=int)
        # equal zero-frequency to anything to escape division by zero
        # we only have frequencies below the Nyquist frequency, period of 1 would map to freq L, which in turn maps to freq 1
        period_map[0:2] = 1
        # we round to closest frequency fitting the period at this index
        period_map[2:] = ((L + 1/2) / periods[2:]).to(int)

        # Spectral density if |fft(autocorr)|^2 / f, we use |a+bi|^2=a^2+b^2
        spect_dens = res.real ** 2 + res.imag ** 2

        # divide by f is equal to multiplying by the period / L, and we just aligned the spectral density to the periods
        corr = corr * spect_dens[:, :, :, period_map] * periods / L

        # time delay agg
        if self.training:
            V = self.time_delay_agg_training(values.permute(0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2)
        else:
            V = self.time_delay_agg_inference(values.permute(0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2)

        if self.output_attention:
            return (V.contiguous(), corr.permute(0, 3, 1, 2))
        else:
            return (V.contiguous(), None)