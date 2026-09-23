# 📖 DevBlog: Memory-Efficient Datasets & Linux Page Cache Optimization

**Author:** Dennis Bob Talanga  
**Category:** MLOps & Performance Engineering  
**Context:** Project *Precipitation Radar Nowcasting using Deep Learning*

---

## The Engineering Challenge: Memory Scalability
The initial dataset implementation loaded and concatenated entire compressed `.npz` radar archives into memory without the use of caching. While straightforward, this approach resulted in massive RAM utilization spikes during dataset construction. Memory consumption scaled linearly (O(N)) with the length of the selected training period, causing system crashes on standard consumer workstations when scaling past a few weeks of data.

## The Production Architecture: OS-Managed Caching & Lazy I/O
The production architecture resolves this bottleneck by completely decoupling application-level memory allocation from dataset size. The system combines **localized sequence indexing**, **lazy loading**, and **NumPy memory mapping (`mmap_mode="r"`)** to keep application memory bounded at O(1) constant overhead while maintaining seamless temporal continuity across archive boundaries.

### Localized Sequence Indexing Pipeline
Instead of keeping the entire raw matrix array resident in memory, the dataset stores only lightweight metadata and sequence indices during initialization. Individual radar archives remain closed on disk until an active training batch requests a sample.

```text
Global dataset
     │
     ├── Archive A
     ├── Archive B
     ├── Archive C
     └── ...
            │
            ▼
  Localized sequence index
(archive_id, frame_id, crosses_boundary)
            │
            ▼
       __getitem__()
            │
            ▼
Memory-map required archive(s) via mmap_mode="r"
            │
            ▼
Extract only required frames -> Concatenate small temporal slices
            │
            ▼
Return training sample / Convert to PyTorch Tensor
```

Each call to `__getitem__()`:
1. Opens a memory map (`mmap_mode="r"`) to the target uncompressed `.npy` archive file.
2. Extracts exclusively the requested temporal frame index slices.
3. Transparently handles cross-archive edge cases if a sequence spans across physical file boundaries.
4. Materializes and concatenates only the localized slices into RAM required for the active batch.
5. Passes the multi-channel grid directly into PyTorch tensors.

### Operating System Page Cache Delegation
Rather than maintaining complex, custom Python-level LRU caching logic across multiple background processes—which introduces worker synchronization overhead—this design deliberately delegates file caching entirely to the **Linux Kernel Page Cache**. 

When a background worker reads an uncompressed chunk from disk via memory mapping, the OS automatically retains those disk blocks in unallocated system RAM. Repeated hits to neighboring temporal windows read directly out of physical memory at hardware speeds without triggering hardware I/O interrupts.

---

## Data Pipeline Benchmarks

To validate the efficiency of this engineering shift, benchmarks were executed on a training split spanning 6 continuous months of high-resolution radar data:

| Configuration | Peak RAM | Mean Epoch Time | Operational Notes |
| :--- | :---: | :---: | :--- |
| **Original In-Memory Loading** | *Scales Linearly* | *N/A (OOM Crash)* | All relevant archives resident; triggered kernel OOM-killer. |
| **Naive Disk Lazy Loading** | **Fixed (Minimal)** | 1205 s | Bounded memory footprint, but heavy performance penalty due to synchronous I/O. |
| **Lazy Loading + Optimized DataLoader** | **Fixed (Minimal)** | **427 s** | **Production Layout:** Overlaps data preparation and GPU computation via pinned memory and prefetching. |

### Overlapping Compute with Parallel Stream Optimization
To ensure the lazy loading pipeline never starves the GPU, the extraction architecture utilizes an optimized PyTorch `DataLoader` configuration:
* `num_workers > 0`: Parallelizes I/O operations across separate background CPU worker processes.
* `pin_memory=True`: Enables direct memory access (DMA) transfers, accelerating host-to-device tensor copies onto the GPU.
* `persistent_workers=True`: Prevents the system from tearing down and recreating CPU worker processes between training epochs, removing lifecycle overhead.
* `prefetch_factor=2`: Forces background workers to aggressively stage the next two batches in memory ahead of the current training step execution loop.
