// ============================================================
// Parakeet Deep Dive — Interactive widgets
// All widgets are vanilla JS + SVG + Plotly. No build step.
// ============================================================

const COLORS = {
  accent: getComputedStyle(document.documentElement).getPropertyValue('--accent').trim(),
  accent2: getComputedStyle(document.documentElement).getPropertyValue('--accent-2').trim(),
  fg: getComputedStyle(document.documentElement).getPropertyValue('--fg').trim(),
  muted: getComputedStyle(document.documentElement).getPropertyValue('--fg-muted').trim(),
  bg: getComputedStyle(document.documentElement).getPropertyValue('--bg').trim(),
};

// ============================================================
// Widget: Alignment-by-hand
// ============================================================
(function alignmentWidget() {
  const root = document.getElementById('align-widget');
  if (!root) return;
  const N = 10;
  const labels = ['_', 'C', 'A', 'T'];
  let cur = Array(N).fill(0);  // all blanks initially

  function collapse(seq) {
    // CTC collapse: remove repeats, then remove blanks
    const merged = [];
    let prev = null;
    for (const s of seq) {
      if (s !== prev) merged.push(s);
      prev = s;
    }
    return merged.filter(s => s !== '_').join('');
  }

  function render() {
    const collapsed = collapse(cur.map(i => labels[i]));
    const ok = collapsed === 'CAT';
    root.innerHTML = `
      <div style="display:flex; gap:6px; margin: 10px 0; flex-wrap:wrap;">
        ${cur.map((v, i) => `
          <div style="text-align:center;">
            <div style="font-size:11px;color:var(--fg-muted);font-family:Inter,sans-serif;">frame ${i}</div>
            <div data-i="${i}" class="afm-cell"
              style="width:42px;height:42px;border:2px solid var(--widget-border);border-radius:6px;
                     display:flex;align-items:center;justify-content:center;font-weight:600;font-size:18px;
                     cursor:pointer; background: ${v===0?'var(--bg-elev)':'var(--accent)'}; color: ${v===0?'var(--fg-muted)':'white'};">
              ${labels[v]}
            </div>
          </div>
        `).join('')}
      </div>
      <div style="font-family: 'Inter', sans-serif; font-size: 14px; margin: 0.8em 0;">
        Per-frame labels: <code>${cur.map(i => labels[i]).join('')}</code><br>
        Collapsed: <code>${collapsed || '(empty)'}</code>
        <span style="margin-left: 1em; color: ${ok ? 'var(--good)' : 'var(--bad)'}; font-weight: 600;">
          ${ok ? '✓ Valid alignment for CAT!' : '✗ Does not match CAT'}
        </span>
      </div>
      <div style="font-size:12px;color:var(--fg-muted);font-family:Inter,sans-serif;">
        Click a cell to cycle through {blank, C, A, T}. CTC's collapse rule: collapse repeats, then strip blanks. e.g. <code>__CCATT_</code> → <code>CAT</code>, <code>CCAATT__</code> → <code>CAT</code>, <code>CTACT___</code> → <code>CTACT</code> (invalid).
      </div>
    `;
    root.querySelectorAll('.afm-cell').forEach(c => {
      c.addEventListener('click', () => {
        const i = +c.dataset.i;
        cur[i] = (cur[i] + 1) % labels.length;
        render();
      });
    });
  }
  render();
})();

// ============================================================
// Widget: Sampling / aliasing
// ============================================================
(function samplingWidget() {
  const root = document.getElementById('sampling-widget');
  if (!root) return;
  root.innerHTML = `
    <div class="controls">
      <label>Sample rate (Hz)<input id="samp-rate" type="range" min="3" max="60" step="1" value="20"><span class="value-display" id="samp-rate-v">20</span></label>
      <label>Signal frequency (Hz)<input id="samp-freq" type="range" min="2" max="20" step="0.5" value="5"><span class="value-display" id="samp-freq-v">5</span></label>
    </div>
    <div id="samp-plot" style="height: 260px;"></div>
    <div id="samp-info" style="font-family: Inter, sans-serif; font-size: 13px; margin-top: 0.5em;"></div>
  `;
  const sr = root.querySelector('#samp-rate');
  const sf = root.querySelector('#samp-freq');
  const srv = root.querySelector('#samp-rate-v');
  const sfv = root.querySelector('#samp-freq-v');
  const info = root.querySelector('#samp-info');

  function draw() {
    const fs = +sr.value;
    const f = +sf.value;
    srv.textContent = fs;
    sfv.textContent = f;
    // Continuous: 0..1 second, dense samples
    const tDense = [];
    const yDense = [];
    for (let i = 0; i <= 500; i++) { const t = i/500; tDense.push(t); yDense.push(Math.sin(2*Math.PI*f*t)); }
    // Discrete samples at chosen rate
    const tSamp = [];
    const ySamp = [];
    for (let i = 0; i <= fs; i++) { const t = i/fs; tSamp.push(t); ySamp.push(Math.sin(2*Math.PI*f*t)); }
    // Reconstruction: cubic interpolation of samples is fine for visualization
    // We'll just draw straight lines between sample points to show what the algorithm "sees"
    const traces = [
      { x: tDense, y: yDense, mode:'lines', name:'Continuous signal',
        line:{color:'#5cc3cc', width:2}, opacity:0.6 },
      { x: tSamp, y: ySamp, mode:'lines', name:'Reconstruction (straight lines)',
        line:{color:'#e29578', width:1.5, dash:'dot'} },
      { x: tSamp, y: ySamp, mode:'markers', name:'Samples',
        marker:{color:'#e29578', size:10} },
    ];
    Plotly.newPlot('samp-plot', traces, {
      margin:{l:40,r:20,t:10,b:40}, paper_bgcolor:'transparent', plot_bgcolor:'transparent',
      font:{color: COLORS.muted, size:11},
      xaxis:{title:'time (s)', gridcolor:'rgba(128,128,128,0.2)'},
      yaxis:{range:[-1.3,1.3], gridcolor:'rgba(128,128,128,0.2)'},
      legend:{orientation:'h', y:-0.25}
    }, {displayModeBar:false, responsive:true});

    const ratio = fs / (2*f);
    const aliased = fs < 2*f;
    info.innerHTML = `<strong>Nyquist rate:</strong> 2 × ${f} = ${2*f} Hz. Your sample rate is ${fs} Hz (${ratio.toFixed(2)}× Nyquist). `
      + (aliased
        ? `<span style="color:var(--bad)">⚠ Below Nyquist — reconstruction is wrong; an aliased frequency of about ${Math.abs(f - Math.round(f/fs)*fs).toFixed(1)} Hz appears instead.</span>`
        : `<span style="color:var(--good)">✓ Above Nyquist — reconstruction is faithful.</span>`);
  }
  sr.addEventListener('input', draw); sf.addEventListener('input', draw); draw();
})();

// ============================================================
// Widget: Fourier builder
// ============================================================
(function fourierWidget() {
  const root = document.getElementById('fourier-widget');
  if (!root) return;
  const comps = [
    { freq: 1, amp: 1.0, phase: 0 },
    { freq: 3, amp: 0.5, phase: 0 },
    { freq: 0, amp: 0.0, phase: 0 },
    { freq: 0, amp: 0.0, phase: 0 },
  ];
  function render() {
    root.innerHTML = `
      <div style="display:grid; grid-template-columns: repeat(4, 1fr); gap: 0.5em;">
        ${comps.map((c,i) => `
          <div style="background:var(--bg); padding:8px; border-radius:6px; border:1px solid var(--widget-border);">
            <div style="font-size:12px;color:var(--fg-muted); font-family:Inter,sans-serif; font-weight:600;">Component ${i+1}</div>
            <label style="font-size:11px;display:block;margin-top:6px;">Freq (Hz)<br><input data-i="${i}" data-k="freq" type="range" min="0" max="15" step="1" value="${c.freq}" style="width:100%"></label>
            <label style="font-size:11px;display:block;">Amp<br><input data-i="${i}" data-k="amp" type="range" min="0" max="1" step="0.05" value="${c.amp}" style="width:100%"></label>
            <label style="font-size:11px;display:block;">Phase<br><input data-i="${i}" data-k="phase" type="range" min="0" max="6.28" step="0.1" value="${c.phase}" style="width:100%"></label>
            <div style="font-family:JetBrains Mono,monospace;font-size:11px;color:var(--accent);margin-top:4px;">f=${c.freq}, a=${c.amp.toFixed(2)}</div>
          </div>`).join('')}
      </div>
      <div id="four-plots" style="display:grid; grid-template-columns: 1fr 1fr; gap:0.8em; margin-top:1em;">
        <div id="four-time" style="height:240px"></div>
        <div id="four-freq" style="height:240px"></div>
      </div>
    `;
    root.querySelectorAll('input').forEach(inp => {
      inp.addEventListener('input', () => {
        const i = +inp.dataset.i; const k = inp.dataset.k; comps[i][k] = +inp.value;
        render();
      });
    });
    const xs = []; const ys = [];
    for (let i = 0; i <= 400; i++) {
      const t = i/400;
      let y = 0;
      for (const c of comps) y += c.amp * Math.sin(2*Math.PI*c.freq*t + c.phase);
      xs.push(t); ys.push(y);
    }
    Plotly.newPlot('four-time', [{x:xs, y:ys, mode:'lines', line:{color:COLORS.accent, width:2}}],
      { title:{text:'Time domain', font:{size:13}}, margin:{l:40,r:10,t:30,b:30}, paper_bgcolor:'transparent', plot_bgcolor:'transparent', font:{color:COLORS.muted, size:10}, xaxis:{title:'t (s)'}, yaxis:{range:[-3,3]}}, {displayModeBar:false});
    const fxs = []; const fys = [];
    for (let f = 0; f <= 16; f++) {
      let a = 0;
      for (const c of comps) if (c.freq === f) a += c.amp;
      fxs.push(f); fys.push(a);
    }
    Plotly.newPlot('four-freq', [{x:fxs, y:fys, type:'bar', marker:{color:COLORS.accent2}}],
      { title:{text:'Frequency domain', font:{size:13}}, margin:{l:40,r:10,t:30,b:30}, paper_bgcolor:'transparent', plot_bgcolor:'transparent', font:{color:COLORS.muted, size:10}, xaxis:{title:'f (Hz)'}, yaxis:{range:[0,1.5]}}, {displayModeBar:false});
  }
  render();
})();

// ============================================================
// Widget: STFT explorer
// ============================================================
(function stftWidget() {
  const root = document.getElementById('stft-widget');
  if (!root) return;
  root.innerHTML = `
    <div class="controls">
      <label>Signal type
        <select id="stft-sig">
          <option value="chirp">Chirp (frequency sweep 1→15 Hz)</option>
          <option value="two">Two tones (3 Hz then 10 Hz)</option>
          <option value="speech">Pseudo-speech (formants modulated)</option>
        </select>
      </label>
      <label>Window size (samples)<input id="stft-win" type="range" min="16" max="256" step="16" value="64"><span class="value-display" id="stft-win-v">64</span></label>
    </div>
    <div id="stft-time" style="height:140px"></div>
    <div id="stft-spec" style="height:240px"></div>
  `;
  const sig = root.querySelector('#stft-sig');
  const win = root.querySelector('#stft-win');
  const winV = root.querySelector('#stft-win-v');

  function genSignal(N, type) {
    const y = new Float32Array(N);
    if (type === 'chirp') {
      for (let n=0;n<N;n++) {
        const t = n/N;
        const f = 1 + 14*t;
        y[n] = Math.sin(2*Math.PI * f * t * 5);
      }
    } else if (type === 'two') {
      for (let n=0;n<N;n++) {
        const t = n/N;
        y[n] = t<0.5 ? Math.sin(2*Math.PI*3*n/N*10) : Math.sin(2*Math.PI*10*n/N*10);
      }
    } else {
      for (let n=0;n<N;n++) {
        const t = n/N;
        // pseudo-speech: two slowly-changing formants
        const f1 = 4 + 2*Math.sin(2*Math.PI*0.5*t);
        const f2 = 9 + 3*Math.sin(2*Math.PI*0.3*t);
        y[n] = 0.5*Math.sin(2*Math.PI*f1*t*10) + 0.4*Math.sin(2*Math.PI*f2*t*10);
      }
    }
    return y;
  }

  // tiny DFT (radix-2 FFT would be faster but N is small)
  function dftMag(x) {
    const N = x.length;
    const out = new Float32Array(N/2);
    for (let k=0;k<N/2;k++) {
      let re = 0, im = 0;
      for (let n=0;n<N;n++) {
        const ang = -2*Math.PI*k*n/N;
        re += x[n]*Math.cos(ang);
        im += x[n]*Math.sin(ang);
      }
      out[k] = Math.sqrt(re*re+im*im);
    }
    return out;
  }

  function draw() {
    const W = +win.value;
    winV.textContent = W;
    const N = 512;
    const y = genSignal(N, sig.value);
    Plotly.newPlot('stft-time',
      [{x: Array.from({length:N}, (_,i)=>i), y: Array.from(y), mode:'lines', line:{color:COLORS.accent, width:1}}],
      { title:{text:'Input signal', font:{size:12}}, margin:{l:40,r:10,t:25,b:20}, paper_bgcolor:'transparent', plot_bgcolor:'transparent', font:{color:COLORS.muted, size:10}, xaxis:{title:'sample n'}, yaxis:{range:[-2,2]} },
      {displayModeBar:false});
    // STFT
    const hop = Math.max(4, W/4);
    const cols = [];
    for (let start=0; start+W<=N; start+=hop) {
      const seg = new Float32Array(W);
      for (let i=0;i<W;i++) {
        const han = 0.5*(1-Math.cos(2*Math.PI*i/(W-1)));
        seg[i] = y[start+i]*han;
      }
      cols.push(Array.from(dftMag(seg)));
    }
    // pad cols to same length
    const maxLen = Math.max(...cols.map(c=>c.length));
    const z = [];
    for (let r=0;r<maxLen;r++) z.push(cols.map(c=>c[r]||0));
    Plotly.newPlot('stft-spec',
      [{z, type:'heatmap', colorscale:'Viridis', showscale:false}],
      { title:{text:`Spectrogram (window=${W} samples)`, font:{size:12}}, margin:{l:40,r:10,t:25,b:30}, paper_bgcolor:'transparent', plot_bgcolor:'transparent', font:{color:COLORS.muted, size:10}, xaxis:{title:'time frame'}, yaxis:{title:'freq bin'} },
      {displayModeBar:false});
  }
  sig.addEventListener('change', draw); win.addEventListener('input', draw); draw();
})();

// ============================================================
// Widget: Mel vs linear filterbank
// ============================================================
(function melWidget() {
  const root = document.getElementById('mel-widget');
  if (!root) return;
  root.innerHTML = `
    <div class="controls">
      <label>Number of filters<input id="mel-n" type="range" min="10" max="80" step="2" value="20"><span class="value-display" id="mel-n-v">20</span></label>
    </div>
    <div id="mel-plots" style="display:grid; grid-template-columns: 1fr 1fr; gap:0.8em;">
      <div id="mel-lin" style="height:240px"></div>
      <div id="mel-mel" style="height:240px"></div>
    </div>
  `;
  const sliderN = root.querySelector('#mel-n');
  const labelN = root.querySelector('#mel-n-v');

  function hzToMel(f) { return 2595 * Math.log10(1 + f/700); }
  function melToHz(m) { return 700 * (Math.pow(10, m/2595) - 1); }

  function triangleBank(centers, fmax, nFFT) {
    const filters = [];
    for (let i=1;i<centers.length-1;i++) {
      const left = centers[i-1], peak = centers[i], right = centers[i+1];
      const f = new Array(nFFT).fill(0);
      for (let k=0;k<nFFT;k++) {
        const freq = k/(nFFT-1)*fmax;
        if (freq < left || freq > right) f[k] = 0;
        else if (freq <= peak) f[k] = (freq-left)/(peak-left);
        else f[k] = (right-freq)/(right-peak);
      }
      filters.push(f);
    }
    return filters;
  }

  function plotBank(divId, title, filters, fmax) {
    const xs = Array.from({length: filters[0].length}, (_,k)=>k/(filters[0].length-1)*fmax);
    const traces = filters.map((f,i) => ({x:xs, y:f, mode:'lines', line:{width:1.2}, showlegend:false}));
    Plotly.newPlot(divId, traces,
      { title:{text:title, font:{size:12}}, margin:{l:40,r:10,t:25,b:35}, paper_bgcolor:'transparent', plot_bgcolor:'transparent', font:{color:COLORS.muted, size:10}, xaxis:{title:'frequency (Hz)'}, yaxis:{range:[0,1.1]} },
      {displayModeBar:false});
  }

  function draw() {
    const n = +sliderN.value; labelN.textContent = n;
    const fmax = 8000;
    const nFFT = 257;
    // linear centres
    const linCenters = Array.from({length:n+2}, (_,i)=> i*fmax/(n+1));
    const linFilters = triangleBank(linCenters, fmax, nFFT);
    // mel centres
    const melMax = hzToMel(fmax);
    const melCenters = Array.from({length:n+2}, (_,i)=> melToHz(i*melMax/(n+1)));
    const melFilters = triangleBank(melCenters, fmax, nFFT);
    plotBank('mel-lin', 'Linear filterbank', linFilters, fmax);
    plotBank('mel-mel', 'Mel filterbank', melFilters, fmax);
  }
  sliderN.addEventListener('input', draw); draw();
})();

// ============================================================
// Widget: SpecAugment
// ============================================================
(function specAugWidget() {
  const root = document.getElementById('specaug-widget');
  if (!root) return;
  const F = 80, T = 200;
  let masks = [];
  // generate a fake spectrogram
  const spec = [];
  for (let f=0;f<F;f++) {
    const row = [];
    for (let t=0;t<T;t++) {
      // simulate a few formant bands & some randomness
      let v = 0.5 * Math.exp(-Math.pow((f-25)/8, 2))
            + 0.3 * Math.exp(-Math.pow((f-50)/10, 2))
            + 0.2 * Math.exp(-Math.pow((f-65)/12, 2));
      v *= (0.7 + 0.6*Math.sin(t/20));
      v += 0.1*Math.random();
      row.push(v);
    }
    spec.push(row);
  }

  function applyMasks() {
    const out = spec.map(r => r.slice());
    for (const m of masks) {
      if (m.type === 'freq') {
        for (let f=m.start; f<m.start+m.width && f<F; f++)
          for (let t=0;t<T;t++) out[f][t] = 0;
      } else {
        for (let t=m.start; t<m.start+m.width && t<T; t++)
          for (let f=0;f<F;f++) out[f][t] = 0;
      }
    }
    return out;
  }

  function render() {
    root.innerHTML = `
      <div class="controls">
        <button id="sa-freq">Add freq mask</button>
        <button id="sa-time">Add time mask</button>
        <button id="sa-clear">Clear</button>
        <span style="font-family:Inter,sans-serif;font-size:12px;color:var(--fg-muted)">${masks.length} mask(s) applied</span>
      </div>
      <div id="sa-plot" style="height:280px"></div>
    `;
    root.querySelector('#sa-freq').onclick = () => {
      const width = Math.floor(10 + Math.random()*17);
      const start = Math.floor(Math.random()*(F-width));
      masks.push({type:'freq', start, width}); render();
    };
    root.querySelector('#sa-time').onclick = () => {
      const width = Math.floor(5 + Math.random()*15);
      const start = Math.floor(Math.random()*(T-width));
      masks.push({type:'time', start, width}); render();
    };
    root.querySelector('#sa-clear').onclick = () => { masks = []; render(); };
    Plotly.newPlot('sa-plot',
      [{z: applyMasks(), type:'heatmap', colorscale:'Viridis', showscale:false}],
      { margin:{l:40,r:10,t:10,b:30}, paper_bgcolor:'transparent', plot_bgcolor:'transparent', font:{color:COLORS.muted, size:10}, xaxis:{title:'time frame'}, yaxis:{title:'mel bin'} },
      {displayModeBar:false});
  }
  render();
})();

// ============================================================
// Widget: 1D convolution
// ============================================================
(function convWidget() {
  const root = document.getElementById('conv-widget');
  if (!root) return;
  root.innerHTML = `
    <div class="controls">
      <label>Kernel size<input id="cv-k" type="range" min="1" max="9" step="2" value="3"><span class="value-display" id="cv-k-v">3</span></label>
      <label>Stride<input id="cv-s" type="range" min="1" max="4" step="1" value="1"><span class="value-display" id="cv-s-v">1</span></label>
      <label>Input length<input id="cv-n" type="range" min="10" max="30" step="1" value="20"><span class="value-display" id="cv-n-v">20</span></label>
    </div>
    <div id="cv-info" style="font-family:Inter,sans-serif;font-size:13px;color:var(--fg-muted); margin-bottom:0.6em;"></div>
    <div id="cv-plot" style="height:240px"></div>
  `;
  const K = root.querySelector('#cv-k'); const S = root.querySelector('#cv-s'); const N = root.querySelector('#cv-n');
  const Kv = root.querySelector('#cv-k-v'); const Sv = root.querySelector('#cv-s-v'); const Nv = root.querySelector('#cv-n-v');
  const info = root.querySelector('#cv-info');
  function draw() {
    const k = +K.value, s = +S.value, n = +N.value;
    Kv.textContent = k; Sv.textContent = s; Nv.textContent = n;
    // generate input
    const x = []; for (let i=0;i<n;i++) x.push(Math.sin(i*0.5) + 0.3*Math.cos(i*1.3));
    // simple kernel: edge detector
    const kernel = []; for (let i=0;i<k;i++) kernel.push(Math.cos(i*Math.PI/(k-1)+0.001));
    // convolution
    const y = [];
    for (let t=0; t+k<=n; t+=s) {
      let v = 0;
      for (let i=0;i<k;i++) v += kernel[i] * x[t+i];
      y.push(v);
    }
    const outLen = y.length;
    info.innerHTML = `Output length: <code>floor((${n} − ${k})/${s}) + 1 = ${outLen}</code>. With stride ${s}, output is ${(outLen/n).toFixed(2)}× the input length.`;
    Plotly.newPlot('cv-plot', [
      { x: Array.from({length:n},(_,i)=>i), y: x, mode:'lines+markers', name:'input', line:{color:COLORS.accent} },
      { x: Array.from({length:outLen},(_,i)=>i*s + (k-1)/2), y, mode:'lines+markers', name:'conv output', line:{color:COLORS.accent2} },
    ], {margin:{l:40,r:10,t:10,b:30}, paper_bgcolor:'transparent', plot_bgcolor:'transparent', font:{color:COLORS.muted, size:11}, legend:{orientation:'h', y:-0.15}}, {displayModeBar:false});
  }
  [K,S,N].forEach(e => e.addEventListener('input', draw)); draw();
})();

// ============================================================
// Widget: Parameter count calculator (conv variants)
// ============================================================
(function paramCalcWidget() {
  const root = document.getElementById('param-calc-widget');
  if (!root) return;
  root.innerHTML = `
    <div class="controls">
      <label>C_in<input id="pc-cin" type="number" min="1" max="2048" value="256"></label>
      <label>C_out<input id="pc-cout" type="number" min="1" max="2048" value="256"></label>
      <label>Kernel K<input id="pc-k" type="number" min="1" max="31" value="9"></label>
    </div>
    <div id="pc-out" style="font-family:Inter,sans-serif;font-size:14px; margin-top:0.6em;"></div>
  `;
  const cin = root.querySelector('#pc-cin'); const cout = root.querySelector('#pc-cout'); const kk = root.querySelector('#pc-k');
  function fmt(n) { return n.toLocaleString(); }
  function draw() {
    const ci = +cin.value, co = +cout.value, k = +kk.value;
    const full = ci*co*k;
    const dw = ci*k;  // depthwise requires Cin == Cout, but we'll show ci*k
    const dwsep = ci*k + ci*co;
    root.querySelector('#pc-out').innerHTML = `
      <table style="width:100%;">
        <tr><th>Variant</th><th>Formula</th><th>Parameters</th><th>vs full</th></tr>
        <tr><td>Standard conv</td><td><code>C_in × C_out × K</code></td><td>${fmt(full)}</td><td>1.0×</td></tr>
        <tr><td>Depthwise (C_in=C_out)</td><td><code>C × K</code></td><td>${fmt(dw)}</td><td>${(dw/full*100).toFixed(2)}%</td></tr>
        <tr><td>DW-separable</td><td><code>C_in × K + C_in × C_out</code></td><td>${fmt(dwsep)}</td><td>${(dwsep/full*100).toFixed(2)}%</td></tr>
      </table>
      <div style="font-size:12px;color:var(--fg-muted);margin-top:0.4em;">FastConformer's subsampling uses depthwise-separable with C_in=80→256, C_out=256, K=9 — about ${((256*9+80*256)/(80*256*9)*100).toFixed(1)}% the cost of the equivalent standard conv.</div>
    `;
  }
  [cin,cout,kk].forEach(e => e.addEventListener('input', draw)); draw();
})();

// ============================================================
// Widget: Attention heatmap
// ============================================================
(function attentionWidget() {
  const root = document.getElementById('attention-widget');
  if (!root) return;
  const tokens = ['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'dog'];
  const T = tokens.length;
  const dk = 16;
  // random Q, K
  function rand() { return Math.random()*2-1; }
  const Q = Array.from({length:T}, () => Array.from({length:dk}, rand));
  const K = Array.from({length:T}, () => Array.from({length:dk}, rand));
  // bias toward attending to neighbours and to "the"
  for (let i=0;i<T;i++) for (let j=0;j<T;j++) {
    if (tokens[j]==='the') K[j][0] += 0.5;
    if (Math.abs(i-j)<=1) Q[i][0] += 0.2;
  }
  function dot(a,b){let s=0;for(let i=0;i<a.length;i++)s+=a[i]*b[i];return s;}
  function softmax(a){const m=Math.max(...a);const e=a.map(x=>Math.exp(x-m));const s=e.reduce((p,c)=>p+c,0);return e.map(x=>x/s);}

  // compute scores
  const scores = [];
  for (let i=0;i<T;i++) {
    const row = [];
    for (let j=0;j<T;j++) row.push(dot(Q[i],K[j])/Math.sqrt(dk));
    scores.push(softmax(row));
  }
  let selected = 0;

  function render() {
    const cellW = 50;
    let html = '<div style="overflow-x:auto"><table style="border-collapse:separate;border-spacing:2px;font-family:Inter,sans-serif;font-size:12px;margin:0 auto;">';
    html += '<tr><th></th>' + tokens.map((t,j) => `<th style="text-align:center;color:var(--fg-muted);font-weight:500;">${t}</th>`).join('') + '</tr>';
    for (let i=0;i<T;i++) {
      html += `<tr><th style="color:${i===selected?'var(--accent)':'var(--fg-muted)'};font-weight:${i===selected?700:500};text-align:right;cursor:pointer;" data-row="${i}">${tokens[i]} ▶</th>`;
      for (let j=0;j<T;j++) {
        const v = scores[i][j];
        const intensity = Math.round(v*255*1.5);
        const bg = `rgba(0, 109, 119, ${v*1.5})`;
        const border = (i===selected) ? 'border:2px solid var(--accent)' : 'border:1px solid var(--widget-border)';
        html += `<td title="i=${i}, j=${j}: ${v.toFixed(3)}" style="width:${cellW}px;height:32px;text-align:center;background:${bg};color:${v>0.3?'white':'var(--fg)'};font-family:JetBrains Mono,monospace;font-size:10px;${border};border-radius:3px;">${v.toFixed(2)}</td>`;
      }
      html += '</tr>';
    }
    html += '</table></div>';
    html += `<div style="font-family:Inter,sans-serif;font-size:13px;margin-top:0.8em;color:var(--fg-muted)">Row ${selected} (<strong>${tokens[selected]}</strong>) attends to: ` + scores[selected].map((v,j)=>`<span style="color:${v>0.2?'var(--accent)':'inherit'}">${tokens[j]}(${(v*100).toFixed(0)}%)</span>`).join(', ') + '</div>';
    html += '<div style="font-size:12px;color:var(--fg-muted);margin-top:0.4em">Click a row label to select a query position. Rows sum to 1.0 (softmax). Darker cells = stronger attention.</div>';
    root.innerHTML = html;
    root.querySelectorAll('[data-row]').forEach(th => {
      th.addEventListener('click', () => { selected = +th.dataset.row; render(); });
    });
  }
  render();
})();

// ============================================================
// Widget: Positional encoding visualisation
// ============================================================
(function peWidget() {
  const root = document.getElementById('pe-widget');
  if (!root) return;
  const T = 100, d = 64;
  const z = [];
  for (let t=0;t<T;t++) {
    const row = [];
    for (let i=0;i<d;i++) {
      const denom = Math.pow(10000, 2*Math.floor(i/2)/d);
      row.push(i%2===0 ? Math.sin(t/denom) : Math.cos(t/denom));
    }
    z.push(row);
  }
  root.innerHTML = '<div id="pe-plot" style="height:280px"></div>';
  Plotly.newPlot('pe-plot',
    [{z, type:'heatmap', colorscale:'RdBu', zmid:0, showscale:true, colorbar:{thickness:8}}],
    {margin:{l:40,r:60,t:10,b:30}, paper_bgcolor:'transparent', plot_bgcolor:'transparent', font:{color:COLORS.muted, size:10}, xaxis:{title:'embedding dimension i'}, yaxis:{title:'position t'}},
    {displayModeBar:false}
  );
})();

// ============================================================
// Widget: Conformer block walkthrough
// ============================================================
(function conformerWidget() {
  const root = document.getElementById('conformer-widget');
  if (!root) return;
  const stages = [
    { name: '½ FFN₁', shape:'[B, T, d]', info:'Macaron half-step feedforward: <code>0.5 × Dropout(W₂·Swish(W₁·LN(x)))</code>. Expand to 4d, activate, project back to d. The half-step (×0.5 on the residual) comes from the Macaron-Net ODE interpretation.' },
    { name:'MHSA (relpos)', shape:'[B, T, d]', info:'Multi-head self-attention with Transformer-XL relative positional encoding. 8 heads, $d_k = d/h$. Each query attends to every key, but the score uses the four-term decomposition (content×content + content×relpos + global-content + global-relpos). This is where global temporal context comes from.' },
    { name:'Conv Module', shape:'[B, T, d]', info:'Pointwise(d→2d) → GLU → DW Conv (k=9) → BatchNorm → Swish → Pointwise(d→d). Provides local context (~720 ms at 80 ms frame rate) that complements the global attention. This is the "conformer" in Conformer.' },
    { name:'½ FFN₂', shape:'[B, T, d]', info:'The second half of the Macaron sandwich. Same as FFN₁: <code>0.5 × Dropout(W₂·Swish(W₁·LN(x)))</code>. Followed by a final LayerNorm.' },
    { name:'LN out', shape:'[B, T, d]', info:'Final LayerNorm. Output shape identical to input — N of these blocks stacked is the encoder body.' },
  ];
  let sel = 0;
  function render() {
    let html = '<div class="pipeline">';
    stages.forEach((s,i) => {
      html += `<div class="pipeline-stage ${i===sel?'active':''}" data-i="${i}"><div class="pipeline-stage-name">${s.name}</div><div class="pipeline-stage-shape">${s.shape}</div></div>`;
      if (i<stages.length-1) html += '<div class="pipeline-arrow">▶</div>';
    });
    html += '</div>';
    html += `<div class="pipeline-detail">${stages[sel].info}</div>`;
    root.innerHTML = html;
    root.querySelectorAll('[data-i]').forEach(el => {
      el.addEventListener('click', () => { sel = +el.dataset.i; render(); });
    });
  }
  render();
})();

// ============================================================
// Widget: Long-form attention patterns
// ============================================================
(function longformWidget() {
  const root = document.getElementById('longform-widget');
  if (!root) return;
  root.innerHTML = `
    <div class="controls">
      <button data-m="full" class="active">Full attention</button>
      <button data-m="window">Sliding window</button>
      <button data-m="windowGlobal">Window + global tokens</button>
      <label>Window radius<input id="lf-w" type="range" min="1" max="20" value="4"><span class="value-display" id="lf-w-v">4</span></label>
      <label>Globals<input id="lf-g" type="range" min="0" max="6" value="2"><span class="value-display" id="lf-g-v">2</span></label>
    </div>
    <div id="lf-canvas" style="text-align:center;"></div>
    <div id="lf-cost" style="font-family:Inter,sans-serif;font-size:13px; margin-top:0.6em;"></div>
  `;
  let mode = 'full';
  const T = 32;
  root.querySelectorAll('button[data-m]').forEach(b => {
    b.addEventListener('click', () => {
      root.querySelectorAll('button[data-m]').forEach(x=>x.classList.remove('active'));
      b.classList.add('active'); mode = b.dataset.m; draw();
    });
  });
  const wEl = root.querySelector('#lf-w'); const gEl = root.querySelector('#lf-g');
  const wV = root.querySelector('#lf-w-v'); const gV = root.querySelector('#lf-g-v');
  wEl.addEventListener('input', draw); gEl.addEventListener('input', draw);

  function draw() {
    const w = +wEl.value, g = +gEl.value;
    wV.textContent = w; gV.textContent = g;
    const cell = 14;
    let svg = `<svg width="${T*cell+60}" height="${T*cell+60}" style="background:var(--bg)">`;
    let computed = 0;
    for (let i=0;i<T;i++) for (let j=0;j<T;j++) {
      let attend = false;
      if (mode==='full') attend = true;
      else if (mode==='window') attend = Math.abs(i-j) <= w;
      else if (mode==='windowGlobal') attend = Math.abs(i-j) <= w || i < g || j < g;
      if (attend) {
        const isGlobal = (mode==='windowGlobal') && (i<g || j<g);
        const color = isGlobal ? COLORS.accent2 : COLORS.accent;
        const alpha = isGlobal ? 0.6 : 0.35;
        svg += `<rect x="${30+j*cell}" y="${30+i*cell}" width="${cell-1}" height="${cell-1}" fill="${color}" opacity="${alpha}" />`;
        computed++;
      } else {
        svg += `<rect x="${30+j*cell}" y="${30+i*cell}" width="${cell-1}" height="${cell-1}" fill="var(--widget-border)" opacity="0.2" />`;
      }
    }
    svg += `<text x="20" y="${30+T*cell/2}" transform="rotate(-90, 20, ${30+T*cell/2})" text-anchor="middle" fill="${COLORS.muted}" font-size="11" font-family="Inter">query i</text>`;
    svg += `<text x="${30+T*cell/2}" y="${30+T*cell+20}" text-anchor="middle" fill="${COLORS.muted}" font-size="11" font-family="Inter">key j</text>`;
    svg += '</svg>';
    root.querySelector('#lf-canvas').innerHTML = svg;
    const full = T*T;
    root.querySelector('#lf-cost').innerHTML = `Computed cells: <span class="value-display">${computed}</span> of ${full} (${(computed/full*100).toFixed(1)}%). For a real 24-min audio (T≈18,000) the savings are dramatic: full = 324M cells, window+global ≈ ${((2*w+1)*18000 + 2*g*18000).toLocaleString()} cells.`;
  }
  draw();
})();

// ============================================================
// Widget: CTC lattice
// ============================================================
(function ctcWidget() {
  const root = document.getElementById('ctc-widget');
  if (!root) return;
  const target = 'CAT';
  const ext = '_C_A_T_';     // extended target: blank-label-blank-...
  const T = 8;
  const S = ext.length;       // 7
  const cellW = 55, cellH = 38, pad = 50;
  const width = T*cellW + pad*2;
  const height = S*cellH + pad*2;
  let path = [];   // [(t, s), ...]

  // simulate per-cell probabilities (just for visual interest)
  const probs = [];
  for (let s=0; s<S; s++) {
    const row = [];
    for (let t=0; t<T; t++) {
      // higher prob along the diagonal
      const target_t = (s/S)*T;
      row.push(Math.exp(-Math.pow((t-target_t)/2, 2)) * 0.8 + 0.1);
    }
    probs.push(row);
  }

  function validNext(t, s, nt, ns) {
    if (nt !== t+1) return false;
    if (ns === s) return true;        // stay
    if (ns === s+1) return true;       // advance
    if (ns === s+2 && ext[s+2] && ext[s+2] !== '_' && ext[s] !== ext[s+2]) return true; // skip blank
    return false;
  }

  function isOnPath(t, s) {
    return path.some(p => p[0]===t && p[1]===s);
  }

  function render() {
    let svg = `<svg class="lattice-svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">`;
    // labels
    for (let s=0; s<S; s++) {
      svg += `<text x="${pad-12}" y="${pad + s*cellH + cellH/2 + 4}" text-anchor="end" font-family="JetBrains Mono" font-size="13" fill="var(--accent)">${ext[s]==='_'?'∅':ext[s]}</text>`;
    }
    for (let t=0; t<T; t++) {
      svg += `<text x="${pad + t*cellW + cellW/2}" y="${pad - 12}" text-anchor="middle" font-family="Inter" font-size="11" fill="var(--fg-muted)">t=${t}</text>`;
    }
    svg += `<text x="20" y="${height/2}" transform="rotate(-90, 20, ${height/2})" text-anchor="middle" font-family="Inter" font-size="12" fill="var(--fg)" font-weight="600">extended target y′</text>`;
    svg += `<text x="${width/2}" y="${height - 8}" text-anchor="middle" font-family="Inter" font-size="12" fill="var(--fg)" font-weight="600">encoder frame t</text>`;

    // edges - show possible transitions from each cell
    for (let t=0; t<T-1; t++) {
      for (let s=0; s<S; s++) {
        const cx1 = pad + t*cellW + cellW/2;
        const cy1 = pad + s*cellH + cellH/2;
        for (let ns of [s, s+1, s+2]) {
          if (ns >= S) continue;
          if (!validNext(t, s, t+1, ns)) continue;
          const cx2 = pad + (t+1)*cellW + cellW/2;
          const cy2 = pad + ns*cellH + cellH/2;
          const onPath = isOnPath(t, s) && isOnPath(t+1, ns);
          svg += `<line x1="${cx1}" y1="${cy1}" x2="${cx2}" y2="${cy2}" class="edge ${onPath?'path':''}" />`;
        }
      }
    }
    // cells
    for (let t=0; t<T; t++) {
      for (let s=0; s<S; s++) {
        const x = pad + t*cellW;
        const y = pad + s*cellH;
        const p = probs[s][t];
        const onPath = isOnPath(t, s);
        const cls = onPath ? 'node active' : 'node';
        const intensity = Math.round(p*100);
        svg += `<rect x="${x+2}" y="${y+2}" width="${cellW-4}" height="${cellH-4}" rx="4" class="${cls}" data-t="${t}" data-s="${s}" fill-opacity="${0.3 + p*0.5}" />`;
        svg += `<text x="${x+cellW/2}" y="${y+cellH/2 + 4}" text-anchor="middle" font-family="JetBrains Mono" font-size="10" fill="${onPath?'white':'var(--fg-muted)'}" pointer-events="none">${p.toFixed(2)}</text>`;
      }
    }
    svg += '</svg>';

    // collapse current path's labels
    const pathLabels = path.map(([t,s]) => ext[s]);
    const collapsed = pathLabels.filter((c,i) => c !== pathLabels[i-1]).filter(c => c !== '_').join('');
    const valid = collapsed === target;

    root.innerHTML = `
      <div style="overflow-x:auto">${svg}</div>
      <div class="controls">
        <button id="ctc-clear">Clear path</button>
        <button id="ctc-rand">Random valid path</button>
        <span style="font-family:Inter,sans-serif;font-size:13px;">
          Path label sequence: <code>${pathLabels.map(c=>c==='_'?'∅':c).join('')}</code>
          → collapses to <code style="color:${valid?'var(--good)':'var(--bad)'}">${collapsed || '(empty)'}</code>
          ${path.length === T ? (valid ? ' ✓' : ' ✗') : ''}
        </span>
      </div>
    `;
    root.querySelectorAll('rect[data-t]').forEach(r => {
      r.addEventListener('click', () => {
        const t = +r.dataset.t, s = +r.dataset.s;
        if (path.length === 0) {
          if (t===0 && (s===0 || s===1)) { path.push([t,s]); render(); }
          return;
        }
        const last = path[path.length-1];
        if (validNext(last[0], last[1], t, s)) { path.push([t,s]); render(); }
      });
    });
    root.querySelector('#ctc-clear').onclick = () => { path=[]; render(); };
    root.querySelector('#ctc-rand').onclick = () => {
      // generate a random valid path
      path = [[0, 0]];
      while (path[path.length-1][0] < T-1) {
        const [t,s] = path[path.length-1];
        const opts = [];
        for (let ns of [s, s+1, s+2]) {
          if (ns >= S) continue;
          if (validNext(t, s, t+1, ns)) opts.push(ns);
        }
        // bias to keep us heading down
        const remaining = T - 1 - t;
        const targetS = Math.min(S-1, s + Math.round((S-1-s) * (1/(remaining+1))));
        const pick = opts.sort((a,b) => Math.abs(a-targetS) - Math.abs(b-targetS))[0];
        path.push([t+1, pick]);
      }
      render();
    };
  }
  render();
})();

// ============================================================
// Widget: RNN-T lattice walker
// ============================================================
(function rnntWidget() {
  const root = document.getElementById('rnnt-widget');
  if (!root) return;
  const T = 8;
  const U = 3;   // target length: C, A, T
  const target = ['C', 'A', 'T'];
  const cellW = 60, cellH = 50, pad = 60;
  const width = (T+1)*cellW + pad*2;
  const height = (U+1)*cellH + pad*2;
  let path = [[0, 0]];

  // simulate alpha values via simple forward
  const alpha = [];
  for (let t=0; t<=T; t++) {
    alpha.push([]);
    for (let u=0; u<=U; u++) {
      alpha[t].push(0);
    }
  }
  alpha[0][0] = 1.0;
  // random uniform transitions for the demo
  const pBlank = 0.5;
  const pToken = 0.5;
  for (let t=0; t<=T; t++) {
    for (let u=0; u<=U; u++) {
      if (t < T) alpha[t+1][u] += alpha[t][u] * pBlank;
      if (u < U) alpha[t][u+1] += alpha[t][u] * pToken;
    }
  }

  function legalNext(t, u, nt, nu) {
    if (nt === t+1 && nu === u) return 'blank';
    if (nt === t && nu === u+1 && u < U) return 'token';
    return null;
  }

  function isOnPath(t, u) {
    return path.some(p => p[0]===t && p[1]===u);
  }

  function render() {
    let svg = `<svg class="lattice-svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">`;
    // axis labels
    svg += `<text x="20" y="${height/2}" transform="rotate(-90, 20, ${height/2})" text-anchor="middle" class="axis-label">output position u →</text>`;
    svg += `<text x="${width/2}" y="${height - 8}" text-anchor="middle" class="axis-label">encoder frame t →</text>`;
    for (let t=0; t<=T; t++) {
      svg += `<text x="${pad + t*cellW + cellW/2}" y="${pad - 12}" text-anchor="middle" font-family="Inter" font-size="11" fill="var(--fg-muted)">${t}</text>`;
    }
    for (let u=0; u<=U; u++) {
      // y axis: u increases downward in our visualisation (more typical for displays)
      const lbl = u === 0 ? '∅' : target[u-1];
      svg += `<text x="${pad-15}" y="${pad + u*cellH + cellH/2 + 4}" text-anchor="end" font-family="JetBrains Mono" font-size="13" fill="var(--accent)">${u}: ${lbl}</text>`;
    }
    // edges - all legal transitions
    for (let t=0; t<=T; t++) {
      for (let u=0; u<=U; u++) {
        const cx1 = pad + t*cellW + cellW/2;
        const cy1 = pad + u*cellH + cellH/2;
        // blank: right
        if (t < T) {
          const cx2 = pad + (t+1)*cellW + cellW/2;
          const onPath = isOnPath(t, u) && isOnPath(t+1, u);
          svg += `<line x1="${cx1}" y1="${cy1}" x2="${cx2}" y2="${cy1}" class="edge ${onPath?'path':''}" />`;
        }
        // token: down
        if (u < U) {
          const cy2 = pad + (u+1)*cellH + cellH/2;
          const onPath = isOnPath(t, u) && isOnPath(t, u+1);
          svg += `<line x1="${cx1}" y1="${cy1}" x2="${cx1}" y2="${cy2}" class="edge ${onPath?'path':''}" />`;
        }
      }
    }
    // cells
    for (let t=0; t<=T; t++) {
      for (let u=0; u<=U; u++) {
        const x = pad + t*cellW;
        const y = pad + u*cellH;
        const onPath = isOnPath(t, u);
        const a = alpha[t][u];
        const cls = onPath ? 'node active' : 'node';
        svg += `<rect x="${x+2}" y="${y+2}" width="${cellW-4}" height="${cellH-4}" rx="4" class="${cls}" data-t="${t}" data-u="${u}" />`;
        svg += `<text x="${x+cellW/2}" y="${y+cellH/2 - 2}" text-anchor="middle" font-family="JetBrains Mono" font-size="10" fill="${onPath?'white':'var(--fg-muted)'}" pointer-events="none">α=${a.toFixed(3)}</text>`;
        svg += `<text x="${x+cellW/2}" y="${y+cellH/2 + 11}" text-anchor="middle" font-family="Inter" font-size="9" fill="${onPath?'white':'var(--fg-faint)'}" pointer-events="none">(${t},${u})</text>`;
      }
    }
    svg += '</svg>';
    // describe path
    const pathDesc = path.length < 2 ? 'start at (0,0); click adjacent cells to extend.' :
      path.slice(0,-1).map((p,i) => {
        const next = path[i+1];
        return legalNext(p[0], p[1], next[0], next[1]) === 'blank' ? '∅' : target[p[1]];
      }).join(' → ');
    const ended = path[path.length-1][0]===T && path[path.length-1][1]===U;
    root.innerHTML = `
      <div style="overflow-x:auto">${svg}</div>
      <div class="controls">
        <button id="rt-clear">Reset</button>
        <button id="rt-rand">Random valid path</button>
        <span style="font-family:Inter,sans-serif;font-size:13px;">Path emissions: <code>${pathDesc}</code> ${ended?'<span style="color:var(--good)">✓ ends at (T,U)</span>':''}</span>
      </div>
    `;
    root.querySelectorAll('rect[data-t]').forEach(r => {
      r.addEventListener('click', () => {
        const t = +r.dataset.t, u = +r.dataset.u;
        const last = path[path.length-1];
        if (legalNext(last[0], last[1], t, u)) { path.push([t,u]); render(); }
      });
    });
    root.querySelector('#rt-clear').onclick = () => { path=[[0,0]]; render(); };
    root.querySelector('#rt-rand').onclick = () => {
      path = [[0,0]];
      while (true) {
        const [t,u] = path[path.length-1];
        if (t===T && u===U) break;
        const opts = [];
        if (t < T) opts.push([t+1, u]);
        if (u < U) opts.push([t, u+1]);
        // bias toward following diagonal
        const target_u = Math.round(U * t/T);
        opts.sort((a,b) => Math.abs(a[1]-target_u) - Math.abs(b[1]-target_u));
        path.push(opts[Math.random()<0.6?0:Math.min(1,opts.length-1)]);
      }
      render();
    };
  }
  render();
})();

// ============================================================
// Widget: TDT lattice with duration jumps
// ============================================================
(function tdtWidget() {
  const root = document.getElementById('tdt-widget');
  if (!root) return;
  const T = 12;
  const U = 3;
  const target = ['C', 'A', 'T'];
  const D = [0, 1, 2, 3, 4];
  const cellW = 50, cellH = 50, pad = 60;
  const width = (T+1)*cellW + pad*2;
  const height = (U+1)*cellH + pad*2;
  let path = [[0, 0]];
  let curD = 1;

  function legalNext(t, u, nt, nu, d) {
    if (!D.includes(d)) return null;
    if (nt === t + d && nu === u + 1) return 'token';
    if (nt === t + d && nu === u && d > 0) return 'blank';
    return null;
  }
  function isOnPath(t, u) { return path.some(p => p[0]===t && p[1]===u); }

  function render() {
    let svg = `<svg class="lattice-svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">`;
    svg += `<text x="20" y="${height/2}" transform="rotate(-90, 20, ${height/2})" text-anchor="middle" class="axis-label">output position u →</text>`;
    svg += `<text x="${width/2}" y="${height - 8}" text-anchor="middle" class="axis-label">encoder frame t (multi-step jumps allowed)</text>`;
    for (let t=0; t<=T; t++) svg += `<text x="${pad + t*cellW + cellW/2}" y="${pad - 12}" text-anchor="middle" font-family="Inter" font-size="11" fill="var(--fg-muted)">${t}</text>`;
    for (let u=0; u<=U; u++) {
      const lbl = u === 0 ? '∅' : target[u-1];
      svg += `<text x="${pad-15}" y="${pad + u*cellH + cellH/2 + 4}" text-anchor="end" font-family="JetBrains Mono" font-size="13" fill="var(--accent)">${u}: ${lbl}</text>`;
    }
    // draw path edges (with d-jumps shown as dashed if d>1)
    for (let i=0; i<path.length-1; i++) {
      const [t1,u1] = path[i]; const [t2,u2] = path[i+1];
      const d = t2 - t1;
      const cx1 = pad + t1*cellW + cellW/2;
      const cy1 = pad + u1*cellH + cellH/2;
      const cx2 = pad + t2*cellW + cellW/2;
      const cy2 = pad + u2*cellH + cellH/2;
      const cls = d>1 ? 'edge duration' : 'edge path';
      svg += `<line x1="${cx1}" y1="${cy1}" x2="${cx2}" y2="${cy2}" class="${cls}" />`;
      // arrow head
      svg += `<circle cx="${cx2}" cy="${cy2}" r="3" fill="var(--accent-2)" />`;
      // label d
      svg += `<text x="${(cx1+cx2)/2}" y="${(cy1+cy2)/2 - 5}" text-anchor="middle" font-family="JetBrains Mono" font-size="10" fill="var(--accent-2)" font-weight="700">d=${d}</text>`;
    }
    // cells
    for (let t=0; t<=T; t++) {
      for (let u=0; u<=U; u++) {
        const x = pad + t*cellW;
        const y = pad + u*cellH;
        const onPath = isOnPath(t, u);
        const cls = onPath ? 'node active' : 'node';
        svg += `<rect x="${x+2}" y="${y+2}" width="${cellW-4}" height="${cellH-4}" rx="4" class="${cls}" data-t="${t}" data-u="${u}" />`;
        svg += `<text x="${x+cellW/2}" y="${y+cellH/2 + 4}" text-anchor="middle" font-family="JetBrains Mono" font-size="10" fill="${onPath?'white':'var(--fg-faint)'}" pointer-events="none">(${t},${u})</text>`;
      }
    }
    svg += '</svg>';

    const lastT = path[path.length-1][0];
    const lastU = path[path.length-1][1];
    const ended = lastT >= T && lastU === U;
    root.innerHTML = `
      <div class="controls">
        <span style="font-family:Inter,sans-serif;font-size:12px;color:var(--fg-muted);">Set jump distance (d):</span>
        ${D.map(d => `<button data-d="${d}" class="${d===curD?'active':''}">${d}</button>`).join('')}
        <button id="tdt-emit-blank">Emit ∅ with d=${curD}</button>
        <button id="tdt-emit-token">Emit ${target[lastU] || '·'} with d=${curD}</button>
        <button id="tdt-clear">Reset</button>
      </div>
      <div style="overflow-x:auto">${svg}</div>
      <div style="font-family:Inter,sans-serif;font-size:13px;margin-top:0.6em;">
        Position: <code>(${lastT}, ${lastU})</code>. Emitted: <code>${path.length-1} step${path.length===2?'':'s'}</code>. ${ended?'<span style="color:var(--good)">✓ reached (T,U)</span>':''}
      </div>
    `;
    root.querySelectorAll('button[data-d]').forEach(b => b.onclick = () => { curD = +b.dataset.d; render(); });
    root.querySelector('#tdt-clear').onclick = () => { path=[[0,0]]; render(); };
    root.querySelector('#tdt-emit-blank').onclick = () => {
      if (curD === 0) return;  // blanks must advance
      const nt = lastT + curD;
      if (nt > T) return;
      path.push([nt, lastU]); render();
    };
    root.querySelector('#tdt-emit-token').onclick = () => {
      if (lastU >= U) return;
      const nt = lastT + curD;
      if (nt > T) return;
      path.push([nt, lastU + 1]); render();
    };
  }
  render();
})();

// ============================================================
// Widget: BPE tokeniser visualiser
// ============================================================
(function bpeWidget() {
  const root = document.getElementById('bpe-widget');
  if (!root) return;
  // hand-crafted toy vocabulary
  const vocab = ['_the', '_quick', '_brown', '_fox', '_jumps', '_over', '_lazy', '_dog',
    '_speech', '_recognition', '_par', '_a', '_an', '_is', 'ake', 'eet', 'tion', 'ing',
    'ed', 'er', 'est', 'ly', 'al', 'or', 'en', 'on', 'at', 'ic', 'ity', 'ous',
    '_re', '_de', '_un', '_pre', '_in', '_to', 'th', 'sh', 'ch', 'ph',
    '_and', '_but', '_or', '_in', '_of', '_for', '_with',
    '_t', '_h', '_e', '_w', '_l', '_y', '_b', '_o', '_d', '_f', '_g', '_k', '_m', '_n', '_p', '_s', '_c'];
  // add all single letters as fallback
  for (let i=0; i<26; i++) vocab.push(String.fromCharCode(97+i));
  // sort by length descending to be greedy with longest match
  vocab.sort((a,b) => b.length - a.length);

  function tokenise(text) {
    text = '_' + text.toLowerCase().replace(/\s+/g, '_');
    const tokens = [];
    let i = 0;
    while (i < text.length) {
      let found = null;
      for (const v of vocab) {
        if (text.substr(i, v.length) === v) { found = v; break; }
      }
      if (!found) { found = text[i]; }
      tokens.push(found);
      i += found.length;
    }
    return tokens;
  }

  root.innerHTML = `
    <div class="controls">
      <input id="bpe-text" type="text" value="the quick brown fox jumps over the lazy dog" style="width:80%;padding:6px;font-family:inherit;border:1px solid var(--widget-border);border-radius:4px;background:var(--bg);color:var(--fg)"/>
    </div>
    <div id="bpe-out" style="font-family:JetBrains Mono,monospace;font-size:13px;margin-top:0.6em;padding:0.8em;background:var(--bg);border-radius:6px;line-height:2.2"></div>
    <div id="bpe-stats" style="font-family:Inter,sans-serif;font-size:12px;color:var(--fg-muted);margin-top:0.4em"></div>
  `;
  function draw() {
    const text = root.querySelector('#bpe-text').value;
    const toks = tokenise(text);
    root.querySelector('#bpe-out').innerHTML = toks.map(t => {
      const color = t.startsWith('_') ? 'var(--accent)' : 'var(--accent-2)';
      return `<span style="background:${color}22;color:${color};padding:3px 6px;border-radius:3px;border:1px solid ${color};margin-right:3px">${t.replace('_','·')}</span>`;
    }).join('');
    root.querySelector('#bpe-stats').innerHTML =
      `Tokens: ${toks.length}. Compression: ${(text.length/toks.length).toFixed(2)} chars per token. ` +
      `Real Parakeet vocab is 1024 tokens; this widget uses ~70.`;
  }
  root.querySelector('#bpe-text').addEventListener('input', draw);
  draw();
})();

// ============================================================
// Widget: Full pipeline explorer
// ============================================================
(function fullPipelineWidget() {
  const root = document.getElementById('full-pipeline-widget');
  if (!root) return;
  const stages = [
    { name:'Audio', shape:'[B, 160k]', info:'Raw 16 kHz mono waveform. 10 seconds = 160,000 float samples. <strong>This is the only input.</strong>'},
    { name:'STFT', shape:'[B, 257, 1000]', info:'Short-time Fourier transform: 25 ms windows, 10 ms hop, n_fft=512 → 257 complex frequency bins per frame. Then take |·|² for power.'},
    { name:'Mel + log + norm', shape:'[B, 80, 1000]', info:'Apply 80-bin mel filterbank, log, per-utterance per-feature normalisation. <strong>Deterministic — no learnable parameters above this line.</strong>'},
    { name:'SpecAugment', shape:'[B, 80, 1000]', info:'Training-only data augmentation: zero out random frequency bands and time stretches.'},
    { name:'Subsampler', shape:'[B, 125, 1024]', info:'Three stride-2 depthwise-separable convs (k=9, 256 channels) → 1024-d linear projection. Output is at 80 ms frame rate. <strong>8× downsampling.</strong>'},
    { name:'24× Conformer blocks', shape:'[B, 125, 1024]', info:'Each block: ½FFN → MHSA(relpos, 8 heads) → ConvModule(k=9, GLU) → ½FFN → LN. Residual + pre-norm around each sub-block. <strong>This is where the heavy lifting happens.</strong>'},
    { name:'Prediction net', shape:'g_u ∈ [1024]', info:'1-layer LSTM over previously emitted tokens. Provides language-model-like context to the joint network. Updated only when a non-blank is emitted.'},
    { name:'Joint network', shape:'[1, |V|+1+|D|]', info:'<code>tanh(W_e·f_t + W_p·g_u + b)</code> → linear projection to 1024 vocab + 1 blank + 5 durations. Two softmax heads: P_T(token) and P_D(duration).'},
    { name:'Greedy TDT decode', shape:'tokens, durations', info:'Loop: query joint(f_t, g_u) → argmax (token, duration) → advance t by duration → if not blank, append token and step prediction net. Stop when t ≥ T.'},
    { name:'Detokenise', shape:'string', info:'Reverse SentencePiece BPE: concatenate tokens, replace underscores with spaces. Final string is what the user sees.'},
  ];
  let sel = 0;
  function render() {
    let html = '<div class="pipeline">';
    stages.forEach((s,i) => {
      html += `<div class="pipeline-stage ${i===sel?'active':''}" data-i="${i}"><div class="pipeline-stage-name">${s.name}</div><div class="pipeline-stage-shape">${s.shape}</div></div>`;
      if (i < stages.length-1) html += '<div class="pipeline-arrow">▶</div>';
    });
    html += '</div>';
    html += `<div class="pipeline-detail"><strong>${stages[sel].name}</strong> &middot; <code>${stages[sel].shape}</code><br>${stages[sel].info}</div>`;
    root.innerHTML = html;
    root.querySelectorAll('[data-i]').forEach(el => el.onclick = () => { sel = +el.dataset.i; render(); });
  }
  render();
})();
