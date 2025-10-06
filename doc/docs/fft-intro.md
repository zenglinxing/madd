Fast Fourier Transform (FFT)
===

Discrete Fourier Transform (DFT) converts the time-domain $T[N]$ signal to the frequency-domain $F[N]$ signal.

$$
F[i] = \sum_{j=0}^{N-1} T[j] \omega^{i\times j}
$$

where $\omega = exp(2\pi i/N)$, which has obviously time complexity as $O(N^{2})$.
Fortunately, Fast Fourier Transform (FFT) acutely simplifies the computation, and reduces the time complexity to $O(N\log N)$.

FFT is a very significant algorithm for signaling and polynomial, etc.
Madd provides FFT functions for general usages on different platforms.

**INVITATIONS!**: I had already considered to call the functions in `FFTW3` library since it has the best optimizations.
However, `FFTW3` applies the GPL open source license, which conflicts with Madd.
Therefore, I have to realize FFT with my limited knowledges.
If you have interest in **optimizing the FFT functions in Madd**, I will be very glad to add your name to the author list.