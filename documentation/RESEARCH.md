# A Real-Time Deep Learning Framework for Vehicle Theft Detection Using Object Detection

## Research foundation for an undergraduate project

**Compiled:** August 2026
**Status:** literature review + research design. Citations gathered via web search and
summarised second-hand — **every reference marked ⚠ must be opened and verified
before it enters your bibliography.** Do not cite a paper you have not read.

---

## 1. Problem framing

### 1.1 The honest motivation

Most undergraduate reports on this topic open with "vehicle theft is rising."
For 2024–2026, in the United States, **that is false**, and an examiner who
knows the numbers will notice.

The National Insurance Crime Bureau reports vehicle theft **fell 23% in 2025**
to 659,880 vehicles — the lowest in decades, and 35.4% below the 2023 peak. The
national rate dropped from 126.62 to 97.33 per 100,000 residents. 49 states
reported declines. ⚠([NICB historic decline](https://www.nicb.org/news/news-releases/us-vehicle-thefts-experience-historic-decline),
[NICB H1 2025](https://www.nicb.org/news/news-releases/nationwide-decline-vehicle-thefts-continues-through-first-half-2025))

So do not argue from a rising trend. Three defensible motivations remain, and
they are stronger because they survive the falling numbers:

**(a) Absolute burden is still large.** 659,880 thefts is not a small number.
NHTSA put annual owner cost above **$8 billion**; the FBI's 2019 per-vehicle
average was **$8,886** across $6.4bn in losses — figures predating recent
vehicle-price inflation.

**(b) The decline is attributable to *prevention*, not detection.** HLDI found
the Hyundai/Kia anti-theft software upgrade cut theft frequency **64%**. That is
an immobiliser fix — a hardening measure. It says nothing about the cases that
still happen, and it is direct evidence that targeted technical intervention
works. Your system addresses a different link in the chain: *detection and
alerting during the attempt.*

**(c) Recovery is where the gap is, and it is time-critical.** California
recovered 84% of stolen passenger vehicles in 2025; Colorado 80%. But
**recreational vehicles: 63% never recovered.** ATVs: only 23% recovered.
Motorcycles: 41%. Critically, **more than half of all recoveries happen within
two weeks, with odds narrowing as time passes.** ⚠([NICB RV report](https://www.nicb.org/news/news-releases/recreational-vehicle-thefts-declined-2025-most-remain-unrecovered-recreational))

**(c) is your strongest argument.** Detection latency directly determines
recovery probability. A system that alerts in seconds rather than hours attacks
the exact variable the recovery data says matters. Frame your contribution
there.

### 1.2 The CCTV evidence gap — your real research justification

Conventional CCTV is passive: it records for post-hoc investigation and relies
on a human watching, which at scale nobody does.

The rigorous evidence for CCTV preventing vehicle theft is **thinner than most
reports assume.** A quasi-experimental study in the *Justice Evaluation Journal*
(2025) used GIS hot-spot mapping and 18 months of pre/post data with weighted
displacement quotient measures, finding burglary/MVT/theft fell **8.5% in hot
spots while control areas rose 13.4%** — a net ~20 percentage point effect, but
**from a single small-city site.** ⚠([Justice Evaluation Journal](https://www.tandfonline.com/doi/full/10.1080/24751979.2025.2474706))

Historically, Campbell Collaboration reviews found CCTV most effective
*specifically in car parks* — which is precisely your deployment context. **Find
and cite that review directly; it is the single most valuable citation for your
introduction and I have not verified it here.**

And the operational failure mode is documented: false alarms are a persistent
problem that **desensitises operators**, which is why the field is moving to AI
classification of person vs. vehicle vs. animal. ⚠([EyeQ Monitoring](https://eyeqmonitoring.com/2025/08/the-real-impact-of-cctv-technology-advancements-on-dealership-theft-prevention/) — trade source, use for
context only, not as evidence)

**This directly validates your threat-scoring design.** Your `-50` owner
suppression rule and the "no single condition alerts alone" invariant are not
implementation conveniences — they are the mechanism addressing the documented
failure mode of the systems you are replacing. Say so explicitly in your
report. It converts an engineering decision into a research contribution.

---

## 2. Literature review

### 2.1 Directly comparable systems

| Work | Approach | Reported result | Note |
|---|---|---|---|
| Real-time theft detection in urban surveillance ⚠ | YOLOv4/v5/v8 + plate recognition + web alerts | YOLOv8: 98% hidden plates, 99% visible, 97% carts | **Closest analogue to your architecture** |
| Optimized hybrid car theft framework ⚠ | Deep transfer learning vs. feature-based ML | survey + comparison | Good related-work source |
| MSAC-SOA (cited within above) ⚠ | Multi-scale attention CNN + improved SSD | 99.88% accuracy | Treat with suspicion — see §2.4 |
| Stolen vehicle surveillance ⚠ | YOLOv7 + DeepSORT + interactive map | — | Tracking-integrated |
| Theft Detection with cGAN + YOLO ⚠ | cGAN vs YOLOv3 vs YOLOv5, weapons | 97.8% / 89.9% / 87.5% | Note: cGAN **beat** YOLO |
| Retail theft classification ⚠ | YOLOv8 | 95% accuracy, 12 ms/frame | Retail, not vehicular |

Sources: [ScienceDirect urban theft](https://www.sciencedirect.com/science/article/pii/S2307187725010375) ·
[Springer hybrid framework](https://link.springer.com/article/10.1007/s10462-025-11480-8) ·
[Springer cGAN+YOLO](https://link.springer.com/content/pdf/10.1007/978-3-031-23724-9_30) ·
[IIETA retail](https://www.iieta.org/journals/isi/paper/10.18280/isi.300610)

### 2.2 The gap you can legitimately claim

Searching specifically for stolen-*vehicle* systems, the literature is
**dominated by traffic management and vehicle counting.** YOLOv8+DeepSORT work
is almost entirely about traffic flow, speed estimation, and counting — not
theft.

The gap is not "nobody has used YOLO for theft." That is false and claiming it
will be caught. **The defensible gap is the specific combination:**

> Plate recognition cross-referenced against an ownership database, **fused with
> face-based owner authorisation**, producing a graduated threat score rather
> than a binary alarm, evaluated with honest edge-hardware latency figures.

The face-recognition-as-authorisation layer is genuinely uncommon in the
vehicle-theft literature. Most work stops at "is this plate on a stolen list."
Your `-50` owner-present suppression has no clear analogue in the papers found.
**That is your contribution — a false-alarm suppression mechanism grounded in
identity, not a novel detector architecture.** Claim that, and nothing more.

### 2.3 Detector backbone: which YOLO, and why

| Version | Key innovation | Trade-off |
|---|---|---|
| YOLOv8 | C2f module | Higher compute cost; less efficient than successors |
| YOLOv9 | PGI + GELAN | Strong accuracy, **slow inference** |
| YOLOv10 | One-to-one head, NMS-free | Fast/small, accuracy lags |
| YOLO11 | C3k2 + C2PSA | 22% fewer params than v8m at ≥ accuracy; ~2% faster than v10 |
| YOLOv12 | Attention-centric | Claims best speed+accuracy; **results inconsistent on edge** |

On vehicle detection specifically: YOLOv8 **45.2%** mAP@[.5:.95], YOLOv10
**46.7%**, YOLOv11 **48.5%**, with v11 notably better on small objects. ⚠([arXiv 2410.22898](https://arxiv.org/pdf/2410.22898))

There is **genuine tension in the literature**: the YOLOv12 paper claims the
speed/accuracy frontier, while applied edge studies still find YOLO11 the best
practical balance, and one MDPI single-board-computer benchmark found **YOLOv12
results inconsistent** on Raspberry Pi. ⚠([arXiv 2411.00201](https://arxiv.org/html/2411.00201v2),
[MDPI SBC benchmark](https://www.mdpi.com/2504-4990/8/7/204))

**Report implication:** your codebase uses YOLOv8n. That is now two generations
old. You have two honest options:

1. **Justify it** — v8n is the most documented, most stable Ultralytics release,
   with the widest deployment literature; reproducibility beats marginal mAP.
2. **Run the comparison** — benchmark v8n vs v11n vs v12n on your own plate
   dataset. This is cheap (same API, one loop) and turns a weakness into a
   contribution: an empirical detector-selection study on *your* task.

**Do option 2 if you have time.** It is the single highest-value addition
available to you, and your `training/evaluate.py` already produces the metrics
and plots. Note the vehicle-detection numbers above predict v11 should win —
if it does, you have replicated a published finding on a new task, which is
exactly what an undergraduate project should demonstrate.

### 2.4 ANPR / plate recognition

Standard pipeline is two-stage: YOLO localises, OCR reads.

| System | Reported |
|---|---|
| YOLOv10 + Tesseract (Thai) | 99.16% detection |
| OCR-YOLOv8 | 98.1% |
| YOLOv7 + PaddleOCR | 97% detect / 95% char |
| YOLOv3-tiny + PyTesseract (edge) | 94% |
| YOLO + OCR (Indian plates) | 87.2% mAP (T4 GPU) |

⚠ Sources: [ScienceDirect YOLO-World ANPR](https://www.sciencedirect.com/science/article/abs/pii/S0045790624005731) ·
[Egyptian YOLOv8+EasyOCR](https://jesit.springeropen.com/articles/10.1186/s43067-024-00156-y) ·
[ETASR constrained devices](https://etasr.com/index.php/ETASR/article/view/9983) ·
[Nature Sci Reports ALPR](https://www.nature.com/articles/s41598-025-24967-9)

**Critical methodological warning, and you must reproduce this reasoning in your
report:** these accuracy figures are **largely self-reported on custom,
non-standardised datasets, so cross-paper comparison is unreliable.** The 87.2%
Indian-plate figure is probably a *harder benchmark*, not worse engineering.
Plate recognition is irreducibly region-specific — performance depends on
country-specific plate design, colour, language, and texture. Applying general
models to region-specific plates remains open.

**CCPD (Chinese City Parking Dataset, 250k images) is the closest thing to a
common benchmark.** If you want a number that is comparable to anything, report
on CCPD. Your EasyOCR choice is well-supported by the Egyptian YOLOv8+EasyOCR
precedent.

Treat any figure above ~99% as a red flag for test-set leakage or a trivially
easy split — including the 99.88% MSAC-SOA claim in §2.1. Saying so in your
report demonstrates exactly the critical reading examiners look for.

### 2.5 Tracking, and why you may need it

Your pipeline has no tracker. Both 2025 YOLOv8+DeepSORT papers flag the same
weakness: **occlusion and ID switches**, with cars showing better associative
stability (AssA) than motorbikes, and larger models improving *recall* without
fixing *identity retention*. ⚠([IJACSA HOTA/CLEAR evaluation](https://thesai.org/Publications/ViewPaper?Volume=16&Issue=9&Code=IJACSA&SerialNo=40),
[ACM CA-BiFPN](https://dl.acm.org/doi/10.1145/3783779.3783831))

For a **single-camera, fixed-view car park**, per-frame detection plus your
existing per-track result caching is defensible — say so and cite the ID-switch
literature as the reason you avoided the added failure mode. For **cross-camera
pursuit**, a dropped ID at handoff means a lost target, and you would need
vehicle re-ID. Scope this out explicitly as future work; do not half-build it.

Note also: your loitering rule (`LOITER_SECONDS`) *is* a temporal feature, and
temporal features imply identity persistence across frames. Be precise in the
report about how you maintain that association without a full tracker — an
examiner will probe exactly there.

### 2.6 Anomaly detection — the road you did not take

**UCF-Crime** (Sultani et al.) is the standard benchmark: 1,900 untrimmed CCTV
videos, 128 hours, 13 anomaly classes — **including "Stealing" and "Burglary."**
Split: 1,610 training videos with video-level labels, 290 test with frame-level.
Metric: frame-level AUC.

Benchmark progression: Sultani (I3D RGB) 76.92 → Zhong (C3D) 81.08 → Wu 82.44 →
MIST 82.30 → CLAWS 83.03 → BN-SVP 83.39 → MSLNet 85.30, with RTFM pushing
further. **UCF Crime2Local** adds spatiotemporal annotation to 300 videos.
⚠([RTFM arXiv 2101.10030](https://arxiv.org/pdf/2101.10030),
[UCF-Crime overview](https://deepwiki.com/wbfwonderful/Fed-WSVAD/3.1-ucf-crime-dataset))

**Note the ceiling: ~85% AUC after years of work by specialist groups.** Compare
that to the 98–99% claims in §2.1 and §2.4. This is the clearest possible
evidence that those numbers reflect easy custom datasets, not superior methods.
**Use this contrast in your report** — it is a genuinely sophisticated
observation and costs you nothing.

Your framework is **rule-based over object detections**, not learned anomaly
detection. That is a legitimate choice for an undergraduate project:
interpretable, debuggable, no large-scale video training, and every alert has a
stated cause. Weakly-supervised VAD is the alternative paradigm — **discuss it
in related work, cite the AUC ceiling, and state why interpretability suited
your deployment context.** Do not pretend you did it.

---

## 3. Reality check on your implementation

Read `README.md` §"Two things to know" — it is already honest, and that honesty
is an asset. Reinforcing it:

### 3.1 The 3.1 FPS finding is your best empirical result

Measured: 320 ms/frame, **3.1 FPS**, yolov8n @ 640px, CPU. Target was 25–30.

**This is not a failure. It is a finding, and the literature backs you up:**

| Platform | Reported |
|---|---|
| Jetson Nano, YOLOv8n CUDA | 163–170 ms ≈ **6 FPS** |
| Jetson Nano, general detectors | **5.2–7.4 FPS** across power modes |
| Jetson Nano, YOLOv5s optimised | 86.2 ms ≈ **9 FPS** end-to-end |
| Jetson Nano, YOLOv4-tiny-288 TensorRT FP16 | **only variant meeting 30 FPS** |
| Orin Nano, YOLOv7-seg FP16 | 42–73 ms ≈ **14–24 FPS** |
| AGX Orin | 18–35 ms ≈ **29–56 FPS** |

⚠([arXiv 2509.13396](https://arxiv.org/pdf/2509.13396),
[arXiv 2307.16834 Jetson benchmarking](https://arxiv.org/pdf/2307.16834),
[arXiv 2602.13378 LAF-YOLOv10](https://arxiv.org/pdf/2602.13378))

**The published consensus is: ~5–10 FPS for standard YOLO on original Jetson
Nano; only tiny/reduced-resolution models with TensorRT FP16 approach 30 FPS;
genuine multi-camera 30 FPS needs Orin-class hardware.**

Your 3.1 FPS on CPU sits exactly where theory predicts, *below* the CUDA Jetson
figures because you have no GPU at all. **You can now write: "the measured
throughput is consistent with published Jetson-class benchmarks, and the
literature indicates 30 FPS is unattainable for this model class without
TensorRT FP16 optimisation and dedicated accelerator hardware."**

That is a far stronger report sentence than any unverified 30 FPS claim. It
reframes the shortfall as a *characterised hardware constraint with citations*
rather than an unmet requirement.

Your `DETECT_EVERY_N_FRAMES` / `OCR_EVERY_N_FRAMES` / `FACE_EVERY_N_FRAMES`
mitigations are the standard response (batch size 1 to minimise latency,
frame skipping, per-track caching). Document the accuracy cost of each
frame-skip setting — that trade-off curve is a publishable figure and you can
generate it today with `evaluate.py`.

**Recommended additions, in order of value per hour spent:**
1. TensorRT or ONNX export benchmark — literature says FP16 is the single
   biggest win, and it is a one-line Ultralytics export.
2. Latency breakdown per stage (detect / OCR / face). Identifies the real
   bottleneck. Cheap to instrument.
3. Accuracy-vs-frame-skip curve. Directly justifies your defaults.

### 3.2 The face backend substitution is fine — document it as a constraint

YuNet + SFace instead of DeepFace/FaceNet, because Python 3.14 has no
TensorFlow wheels. SFace reports ~99.6% on LFW. Your `/api/info` reporting the
active backend so a demo cannot misrepresent results is genuinely good practice
— **mention that design decision explicitly, it demonstrates research
integrity.**

### 3.3 The gap nobody has flagged yet: you have no dataset

`dataset/raw/` and `dataset/annotations/` are **both empty.** `models/license_plate.pt`
does not exist. The system is running on the **classical OpenCV edge-density
fallback**, not a trained plate detector.

**This is the largest hole in the project.** Consequences:

- No trained plate model means no mAP, precision, or recall for plate
  localisation — the numbers a marker will look for first.
- The README's "Training & evaluation pipeline: Complete" means the *code* is
  complete. **Nothing has been trained.** Ensure your report never implies
  otherwise; that distinction is exactly what a viva probes.
- `evaluate.py` currently benchmarks `yolov8n.pt` (COCO weights), not a
  task-specific model.

**This is the highest priority remaining work, ahead of everything in §3.1.**
Options, cheapest first: CCPD (250k images, gives comparability), an open
Roboflow plate dataset, or hand-annotating a few hundred local images —
region-specific per §2.4, and enough for a genuine result.

Without it you have a well-engineered system with no core empirical result. With
it, plus the §2.3 detector comparison, you have a complete project.

---

## 4. Ethics, privacy, and legal framing

Do not treat this as a box-tick. Your README §"Security and ethics" is a good
start; the literature gives it teeth.

**Biometric data is a special category.** GDPR defines biometric data as
personal data from technical processing of physical/physiological/behavioural
characteristics, and **facial images qualify** because they identify
individuals. Explicit consent is required; individuals have a right to object.
Scholars note GDPR provisions on consent and data minimisation **may not
adequately address meaningful consent in public-space surveillance.**
⚠([Springer AFR governance](https://link.springer.com/article/10.1365/s43439-021-00022-x),
[Frontiers privacy/ethics](https://www.frontiersin.org/journals/big-data/articles/10.3389/fdata.2024.1337465/full))

**Case law:** *Peck v. United Kingdom* (ECtHR) held that video surveillance of
public places where data is recorded, stored, and disclosed falls within
Article 8. Courts increasingly limit AFR surveillance.

**Regulatory:** Oregon and New Hampshire banned FR in police body cameras; some
jurisdictions bar shops, gyms, and public transport from FR in video
surveillance with GDPR-scale fines. **Follow up: EU AI Act Article 5
prohibitions on real-time remote biometric identification, and EDPB Guidelines
3/2019 on video devices** — both directly govern a system like yours and neither
is verified here.

**Mandate creep** is the risk to name. The *Amadeus* paper frames it precisely:
a stream processed by multiple parties, each with its own mandate, where **a
consumer may exceed their mandate by performing facial recognition without the
administrator's knowledge.** ⚠([arXiv 2011.05163](https://arxiv.org/pdf/2011.05163))
Your system stores face embeddings and captures photographs of unregistered
people — exactly the capability that invites creep.

**A useful precedent for how to write your ethics statement:** the *uxSense*
authors openly address dual-use by stating their prototype *omits* facial
recognition while acknowledging its field utility. You cannot omit it, so
instead: state the capability, state the safeguard, state the residual risk.

**Concrete actions before submission:**
- Institutional ethics approval — likely required, check early, it has lead time.
- A stated retention period, implemented, not just described.
- Fix `/media` being unauthenticated. It serves identifiable faces and plates.
  Your README already flags this; **an examiner reading that will ask why it is
  still open.** Fix it or justify it as scoped-out with a documented risk.
- Document the MJPEG query-string token as a known, reasoned trade-off. You
  already do — keep it.
- Mitigations worth citing as future work: homomorphic encryption, secure
  multiparty computation, synthetic training data, privacy gain / attribute
  suppression rate as metrics. ⚠([ACM PPFR survey](https://dl.acm.org/doi/10.1145/3673224),
  [Springer privacy survey](https://link.springer.com/article/10.1007/s42452-025-06987-2))

---

## 5. Suggested report structure

1. **Introduction** — burden (§1.1a), prevention-vs-detection (§1.1b),
   **recovery time-criticality as the core argument (§1.1c)**. Aims, objectives,
   scope.
2. **Literature review** — CCTV evidence and the false-alarm/desensitisation
   failure mode (§1.2); comparable systems (§2.1); YOLO evolution (§2.3); ANPR
   and its benchmarking problem (§2.4); tracking and ID switches (§2.5);
   weakly-supervised VAD and its ~85% AUC ceiling as contrast (§2.6). Close with
   the gap statement (§2.2).
3. **Methodology** — architecture (README §Architecture), threat scoring as
   false-alarm mitigation, dataset preparation, training protocol, metrics
   (mAP@.5, mAP@[.5:.95], precision, recall, latency, FPS).
4. **Implementation** — modules, the layered import discipline, the face-backend
   substitution as a documented constraint (§3.2).
5. **Results** — plate detector metrics (**§3.3 — must be generated**), detector
   comparison (§2.3 option 2), latency breakdown and FPS against published
   Jetson figures (§3.1), threat-score behaviour, the 79 passing tests.
6. **Discussion** — accuracy/latency trade-off, why 3.1 FPS is consistent with
   the literature, false-alarm suppression as the contribution, limitations.
7. **Ethics** — §4, as a full chapter, not an appendix.
8. **Conclusion & future work** — TensorRT/edge deployment, DeepSORT for
   cross-camera re-ID, learned anomaly detection, privacy-preserving embeddings.

---

## 6. Priority actions

| # | Action | Why | Effort |
|---|---|---|---|
| 1 | **Acquire/annotate a plate dataset and train `license_plate.pt`** | No core empirical result without it (§3.3) | High |
| 2 | Verify every ⚠ citation; find the Campbell CCTV car-park review | Unread citations are the fastest way to fail a viva | Medium |
| 3 | Benchmark v8n vs v11n vs v12n on your data | Turns an outdated-backbone weakness into a contribution (§2.3) | Low |
| 4 | Latency breakdown per stage + TensorRT/ONNX export | Strongest §3.1 additions | Low |
| 5 | Accuracy-vs-frame-skip curve | Justifies your defaults empirically | Low |
| 6 | Start institutional ethics approval | Lead time; may block submission | Low but urgent |
| 7 | Authenticate `/media` | Known flaw, already documented, will be asked about | Low |

**The pattern to notice:** items 3–5 are all low-effort and high-value because
`training/evaluate.py` already exists. Item 1 is the only expensive one, and
it is also the only one you cannot submit without.

---

## 7. Citation integrity — read this before writing

Everything marked ⚠ came from **search-result summaries, not from reading the
papers.** Snippets misattribute numbers, conflate a paper's claims with its
related-work section, and occasionally cite work that does not exist.

Specific cautions:

- The MSAC-SOA 99.88% and the YOLOv7+DeepSORT stolen-vehicle system were found
  **cited inside another paper's related-work section.** Trace them to their
  originals; do not cite second-hand.
- The Campbell Collaboration CCTV review is **the most valuable citation for
  your introduction and is unverified.** Find it first.
- Some URLs above have future-looking arXiv identifiers. **Confirm each resolves
  to a real paper** before citing.
- EU AI Act Art. 5 and EDPB Guidelines 3/2019 are named from a search
  suggestion, not verified. Read the primary instruments.

For each reference: open it, confirm the number, confirm it says what you think,
record the full citation. Where a figure is self-reported on a custom dataset —
which per §2.4 is most of them — **say so.** Noting that comparison across these
papers is unreliable is not a weakness in your review; it is the single clearest
signal of critical reading you can give a marker, and §2.6's ~85% AUC ceiling
gives you the evidence to back it up.
