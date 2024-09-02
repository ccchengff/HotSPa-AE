#include "hetu/impl/memory/CUDABFCMemoryPool.cuh"
// #include "hetu/impl/utils/cuda_utils.h"
// #include <mutex>
// namespace hetu { namespace impl {

// constexpr CUDABFCMemoryPool::ChunkHandle CUDABFCMemoryPool::kInvalidChunkHandle;
// CUDABFCMemoryPool::CUDABFCMemoryPool(DeviceIndex device_id, size_t total_memory,
//                                      const std::string &name, bool allow_growth,
//                                      bool garbage_collection,
//                                      double fragmentation_fraction) :
//     MemoryPool(),
//     _device_id(device_id), _total_memory(total_memory), _name(name),
//     _allow_growth(allow_growth), _garbage_collection(garbage_collection),
//     _fragmentation_fraction(fragmentation_fraction),
//     _free_chunks_list(kInvalidChunkHandle), next_allocation_id_(1) {

//     if (allow_growth) {
//         curr_region_allocation_bytes_ =
//             RoundedBytes(std::min(total_memory, size_t{2 << 20}));
//     } else {
//         curr_region_allocation_bytes_ = RoundedBytes(total_memory);
//     }
//     // Allocate the requested amount of memory.
//     memory_limit_ = total_memory;
//     // HT_LOG_INFO << "Creating new CUDABFCMemoryPool named: " << name;

//     for (BinNum b = 0; b < kNumBins; b++) {
//         size_t bin_size = BinNumToSize(b);
//         new (BinFromIndex(b)) Bin(this, bin_size);
//         HT_ASSERT_EQ(BinForSize(bin_size), BinFromIndex(b));
//         HT_ASSERT_EQ(BinForSize(bin_size + 255), BinFromIndex(b));
//         HT_ASSERT_EQ(BinForSize(bin_size * 2 - 1), BinFromIndex(b));
//         if (b + 1 < kNumBins) {
//             HT_ASSERT_NE(BinForSize(bin_size * 2), BinFromIndex(b));
//         }
//     }
// }

// CUDABFCMemoryPool::~CUDABFCMemoryPool() {
//     // HT_LOG_INFO << "Number of regions allocated: "
//     //             << _region_manager.regions().size();
//     try
//     {
//       DeviceIndex prev_id = SetDevice();
//       for (const auto &region : _region_manager.regions()) {
//           CudaFree(region.ptr());
//       }
//       ResetDevice(prev_id);
//       for (BinNum b = 0; b < kNumBins; b++) {
//           BinFromIndex(b)->~Bin();
//       }
//     }
//     catch(const std::exception& e)
//     {
//       HT_LOG_DEBUG << "In ~CUDABFCMemoryPool(), catch exception: " << e.what();
//     }
// }

// CUDABFCMemoryPool::Chunk *CUDABFCMemoryPool::ChunkFromHandle(ChunkHandle h) {
//     HT_ASSERT_GE(h, 0);
//     HT_ASSERT_LT(h, static_cast<int>(_chunks.size()));
//     return &(_chunks[h]);
// }

// const CUDABFCMemoryPool::Chunk *
// CUDABFCMemoryPool::ChunkFromHandle(ChunkHandle h) const {
//     HT_ASSERT_GE(h, 0);
//     HT_ASSERT_LT(h, static_cast<int>(_chunks.size()));
//     return &(_chunks[h]);
// }

// bool CUDABFCMemoryPool::Extend(size_t rounded_bytes) {
//     size_t available_bytes = memory_limit_ - total_region_allocated_bytes_;
//     // Rounds available_bytes down to the nearest multiple of
//     // kMinAllocationSize.
//     available_bytes =
//         (available_bytes / kMinAllocationSize) * kMinAllocationSize;

//     if (rounded_bytes > available_bytes) {
//         return false;
//     }

//     // If curr_region_allocation_bytes_ is not enough to satisfy the
//     // allocation, keep multiplying by a power of two until that is
//     // sufficient.
//     bool increased_allocation = false;
//     while (rounded_bytes > curr_region_allocation_bytes_) {
//         curr_region_allocation_bytes_ *= 2;
//         increased_allocation = true;
//     }

//     // Try allocating.
//     auto alignment = get_data_alignment();
//     size_t bytes = std::min(curr_region_allocation_bytes_, available_bytes);
//     size_t bytes_received = (bytes + alignment - 1) / alignment * alignment;

//     size_t freeMem;
//     size_t totalMem;
//     cudaMemGetInfo(&freeMem, &totalMem);

//     if (freeMem < rounded_bytes) {
//         return false;
//     } else {
//         bytes_received = std::min(freeMem, bytes_received);
//     }
//     DeviceIndex prev_id = SetDevice();
//     void *mem_addr = nullptr;
//     CudaMalloc(&mem_addr, bytes_received);
//     ResetDevice(prev_id);
//     if (!increased_allocation) {
//         // Increase the region size of the next required allocation.
//         curr_region_allocation_bytes_ *= 2;
//     }

//     // HT_LOG_INFO << "Extending allocation by " << std::to_string(bytes_received)
//     //             << " bytes for " << Name() << ".";

//     total_region_allocated_bytes_ += bytes_received;
//     // HT_LOG_INFO << "Total allocated bytes: "
//     //             << std::to_string(total_region_allocated_bytes_);

//     // HT_LOG_INFO << "Allocated memory at " << mem_addr << " to "
//     //             << static_cast<void *>(static_cast<char *>(mem_addr)
//     //                                    + bytes_received);

//     AllocationRegion *maybe_extended_region = nullptr;
//     // if (coalesce_regions_) {
//     //     maybe_extended_region = region_manager_.AddOrExtendAllocationRegion(
//     //         mem_addr, bytes_received);
//     // } else {
//     _region_manager.AddAllocationRegion(mem_addr, bytes_received);
//     // }
//     // Create one large chunk for the whole memory space that will
//     // be chunked later.
//     ChunkHandle h = AllocateChunk();
//     CUDABFCMemoryPool::Chunk *c = ChunkFromHandle(h);
//     c->ptr = mem_addr;
//     c->size = bytes_received;
//     c->allocation_id = -1;
//     c->prev = kInvalidChunkHandle;
//     c->next = kInvalidChunkHandle;
//     c->freed_at_count = 0;

//     _region_manager.set_handle(c->ptr, h);

//     // If the region was extended, then there exists a previous chunk that
//     // should be linked to the new chunk.
//     if (maybe_extended_region != nullptr) {
//         ChunkHandle prev =
//             maybe_extended_region->get_handle(maybe_extended_region->ptr());
//         CUDABFCMemoryPool::Chunk *prev_chunk = ChunkFromHandle(prev);
//         // Find the last recorded chunk in the extended region.
//         while (prev_chunk->next != kInvalidChunkHandle) {
//             prev = prev_chunk->next;
//             prev_chunk = ChunkFromHandle(prev);
//         }
//         c->prev = prev;
//         prev_chunk->next = h;
//     }

//     // Maybe merge adjacent chunks and insert the chunk into the right bin.
//     InsertFreeChunkIntoBin(TryToCoalesce(h, /*ignore_freed_at=*/false));

//     return true;
// }

// CUDABFCMemoryPool::ChunkHandle CUDABFCMemoryPool::AllocateChunk() {
//     if (_free_chunks_list != kInvalidChunkHandle) {
//         ChunkHandle h = _free_chunks_list;
//         Chunk *c = ChunkFromHandle(h);
//         _free_chunks_list = c->next;
//         return h;
//     } else {
//         ChunkHandle h = _chunks.size();
//         _chunks.resize(h + 1);
//         return h;
//     }
// }

// void CUDABFCMemoryPool::DeallocateChunk(ChunkHandle h) {
//     Chunk *c = ChunkFromHandle(h);
//     c->allocation_id = -1;
//     c->bin_num = kInvalidBinNum;
//     c->next = _free_chunks_list;
//     _free_chunks_list = h;
// }

// DataPtr CUDABFCMemoryPool::AllocDataSpace(size_t num_bytes) {
//     std::lock_guard<std::mutex> lock(_mtx);
//     // HT_LOG_INFO << "AllocateRaw " << Name() << "  " << num_bytes;
//     void *result = AllocateRawInternal(num_bytes);
//     if (result == nullptr) {
//         // HT_LOG_WARN << "Allocator (" << Name() << ") ran out of memory trying "
//         //             << "to allocate " << std::to_string(num_bytes);
//     }
//     // HT_LOG_INFO << "AllocateRaw " << Name() << "  " << num_bytes << " "
//     //             << result;
//     return {result, num_bytes, Device(kCUDA, _device_id)};
//     // return result;
// }

// // static
// size_t CUDABFCMemoryPool::RoundedBytes(size_t bytes) {
//     size_t rounded_bytes =
//         (kMinAllocationSize
//          * ((bytes + kMinAllocationSize - 1) / kMinAllocationSize));
//     HT_ASSERT_EQ(size_t{0}, rounded_bytes % kMinAllocationSize);
//     return rounded_bytes;
// }

// bool CUDABFCMemoryPool::DeallocateFreeRegions(size_t rounded_bytes) {
//     // Do nothing if garbage collection is off.
//     if (!_garbage_collection) {
//         return false;
//     }

//     // Searching for free regions.
//     std::unordered_set<void *> free_region_ptrs;
//     size_t total_free_bytes = 0;
//     for (const AllocationRegion &region : _region_manager.regions()) {
//         ChunkHandle h = _region_manager.get_handle(region.ptr());
//         bool any_use = false;
//         while (h != kInvalidChunkHandle) {
//             const Chunk *c = ChunkFromHandle(h);
//             if (c->is_used()) {
//                 any_use = true;
//                 break;
//             }
//             h = c->next;
//         }

//         if (!any_use) {
//             // HT_LOG_INFO << "Found free region with ptr = " << region.ptr();
//             free_region_ptrs.insert(region.ptr());
//             total_free_bytes += region.memory_size();
//         }
//     }

//     if (total_free_bytes == 0) {
//         return false;
//     }

//     // Rough estimation to check whether deallocation can help.
//     size_t available_bytes =
//         memory_limit_ - total_region_allocated_bytes_ + total_free_bytes;
//     if (rounded_bytes > available_bytes) {
//         return false;
//     }

//     HT_LOG_WARN << "Garbage collection: deallocate free memory regions"
//                 << " (i.e., allocations) so that we can re-allocate a larger"
//                 << " region to avoid OOM due to memory fragmentation.";

//     // Deallocate free regions.
//     DeallocateRegions(free_region_ptrs);
//     return true;
// }

// void CUDABFCMemoryPool::DeallocateRegions(
//     const std::unordered_set<void *> &region_ptrs) {
//     // Explicitly remove the const qualifier as some compilers disallow passing
//     // const_iterator to std::vector::erase(), which is used in
//     // RemoveAllocationRegion().
//     auto regions =
//         const_cast<std::vector<AllocationRegion> *>(&_region_manager.regions());
//     auto it = regions->begin();
//     while (it != regions->end()) {
//         if (region_ptrs.find(it->ptr()) == region_ptrs.end()) {
//             ++it;
//             continue;
//         }
//         // HT_LOG_INFO << "Deallocate region with ptr = " << it->ptr();
//         // Remove all chunk registrations from Bins.
//         ChunkHandle h = _region_manager.get_handle(it->ptr());
//         while (h != kInvalidChunkHandle) {
//             const Chunk *c = ChunkFromHandle(h);
//             if (c->bin_num != kInvalidBinNum) {
//                 RemoveFreeChunkFromBin(h);
//             }
//             auto h_to_delete = h;
//             h = c->next;
//             DeleteChunk(h_to_delete);
//         }
//         DeviceIndex prev_id = SetDevice();
//         // Deallocate the memory.
//         CudaFree(it->ptr());
//         ResetDevice(prev_id);
//         total_region_allocated_bytes_ -= it->memory_size();
//         it = _region_manager.RemoveAllocationRegion(it);
//     }
// }

// void *CUDABFCMemoryPool::AllocateRawInternal(size_t num_bytes) {
//     if (num_bytes == 0) {
//         // HT_LOG_INFO << "tried to allocate 0 bytes";
//         return nullptr;
//     }
//     size_t rounded_bytes = RoundedBytes(num_bytes);

//     // The BFC allocator tries to find the best fit first.
//     BinNum bin_num = BinNumForSize(rounded_bytes);
//     void *ptr = FindChunkPtr(bin_num, rounded_bytes, num_bytes);
//     if (ptr != nullptr) {
//         return ptr;
//     }
//     // Try to extend
//     if (Extend(rounded_bytes)) {
//         ptr = FindChunkPtr(bin_num, rounded_bytes, num_bytes);
//         if (ptr != nullptr) {
//             return ptr;
//         }
//     }
//     // No chunks can satisfy the request. Try deallocating free regions.
//     if (DeallocateFreeRegions(rounded_bytes) && Extend(rounded_bytes)) {
//         ptr = FindChunkPtr(bin_num, rounded_bytes, num_bytes);
//         if (ptr != nullptr) {
//             return ptr;
//         }
//     }
//     return nullptr;
// }

// // int64_t CUDABFCMemoryPool::LargestFreeChunk() {
// //     for (int i = kNumBins - 1; i >= 0; i--) {
// //         if (!BinFromIndex(i)->free_chunks.empty()) {
// //             return ChunkFromHandle(*BinFromIndex(i)->free_chunks.rbegin())
// //                 ->size;
// //         }
// //     }
// //     return 0;
// // }

// void *CUDABFCMemoryPool::FindChunkPtr(BinNum bin_num, size_t rounded_bytes,
//                                       size_t num_bytes) {
//     // First identify the first bin that could satisfy rounded_bytes.
//     for (; bin_num < kNumBins; bin_num++) {
//         // Start searching from the first bin for the smallest chunk that fits
//         // rounded_bytes.
//         Bin *b = BinFromIndex(bin_num);
//         for (auto citer = b->free_chunks.begin(); citer != b->free_chunks.end();
//              ++citer) {
//             const CUDABFCMemoryPool::ChunkHandle h = (*citer);
//             CUDABFCMemoryPool::Chunk *chunk = ChunkFromHandle(h);
//             HT_ASSERT_EQ(chunk->is_used(), false);
//             // if (freed_before > 0 && freed_before < chunk->freed_at_count) {
//             //     continue;
//             // }
//             if (chunk->size >= rounded_bytes) {
//                 // We found an existing chunk that fits us that wasn't in use,
//                 // so remove it from the free bin structure prior to using.
//                 RemoveFreeChunkIterFromBin(&b->free_chunks, citer);

//                 // If we can break the size of the chunk into two reasonably
//                 // large pieces, do don't waste more than
//                 // max_internal_fragmentation_bytes on padding. If this
//                 // threshold is not set by the user, then use 128MB as the
//                 // default.
//                 const int64_t max_internal_fragmentation_bytes =
//                     (_fragmentation_fraction > 0.0) ?
//                         _fragmentation_fraction * memory_limit_ :
//                         128 << 20;

//                 if (chunk->size >= rounded_bytes * 2
//                     || static_cast<int64_t>(chunk->size) - rounded_bytes
//                            >= max_internal_fragmentation_bytes) {
//                     SplitChunk(h, rounded_bytes);
//                     chunk = ChunkFromHandle(
//                         h); // Update chunk pointer in case it moved
//                 }

//                 // The requested size of the returned chunk is what the user
//                 // has allocated.
//                 chunk->requested_size = num_bytes;
//                 // Assign a unique id and increment the id counter, marking the
//                 // chunk as being in use.
//                 chunk->allocation_id = next_allocation_id_++;

//                 // HT_LOG_INFO << "Returning: " << chunk->ptr;
//                 return chunk->ptr;
//             }
//         }
//     }
//     return nullptr;
// }

// void CUDABFCMemoryPool::SplitChunk(CUDABFCMemoryPool::ChunkHandle h,
//                                    size_t num_bytes) {
//     // Allocate the new chunk before we do any ChunkFromHandle
//     ChunkHandle h_new_chunk = AllocateChunk();

//     Chunk *c = ChunkFromHandle(h);
//     HT_ASSERT_EQ(!c->is_used() && (c->bin_num == kInvalidBinNum), true);

//     // Create a new chunk starting num_bytes after c
//     CUDABFCMemoryPool::Chunk *new_chunk = ChunkFromHandle(h_new_chunk);
//     new_chunk->ptr =
//         static_cast<void *>(static_cast<char *>(c->ptr) + num_bytes);
//     _region_manager.set_handle(new_chunk->ptr, h_new_chunk);

//     // Set the new sizes of the chunks.
//     new_chunk->size = c->size - num_bytes;
//     c->size = num_bytes;

//     // The new chunk is not in use.
//     new_chunk->allocation_id = -1;

//     // It inherits the freed time.
//     new_chunk->freed_at_count = c->freed_at_count;

//     // Maintain the pointers.
//     // c <-> c_neighbor becomes
//     // c <-> new_chunk <-> c_neighbor
//     CUDABFCMemoryPool::ChunkHandle h_neighbor = c->next;
//     new_chunk->prev = h;
//     new_chunk->next = h_neighbor;
//     c->next = h_new_chunk;
//     if (h_neighbor != kInvalidChunkHandle) {
//         Chunk *c_neighbor = ChunkFromHandle(h_neighbor);
//         c_neighbor->prev = h_new_chunk;
//     }

//     // Add the newly free chunk to the free bin.
//     InsertFreeChunkIntoBin(h_new_chunk);
// }

// void CUDABFCMemoryPool::FreeDataSpace(DataPtr ptr) {
//     std::lock_guard<std::mutex> lock(_mtx);
//     // HT_LOG_INFO << "DeallocateRaw " << Name() << " "
//                 // << (ptr.ptr ? RequestedSize(ptr.ptr) : 0);
//     DeallocateRawInternal(ptr.ptr);
// }

// void CUDABFCMemoryPool::DeallocateRawInternal(void *ptr) {
//     if (ptr == nullptr) {
//         // HT_LOG_WARN << "tried to deallocate nullptr";
//         return;
//     }

//     // Find the chunk from the ptr.
//     CUDABFCMemoryPool::ChunkHandle h = _region_manager.get_handle(ptr);
//     HT_ASSERT_NE(h, kInvalidChunkHandle);
//     // Record chunk information before it's freed.
//     // Chunk *chunk = ChunkFromHandle(h);
//     // void *chunk_ptr = chunk->ptr;
//     // int64_t req_bytes = chunk->requested_size;
//     // int64_t alloc_bytes = chunk->size;

//     MarkFree(h);
//     InsertFreeChunkIntoBin(TryToCoalesce(h, false));
// }

// // Merges h1 and h2 when Chunk(h1)->next is h2 and Chunk(h2)->prev is c1.
// // We merge Chunk(h2) into Chunk(h1).
// void CUDABFCMemoryPool::Merge(CUDABFCMemoryPool::ChunkHandle h1,
//                               CUDABFCMemoryPool::ChunkHandle h2) {
//     Chunk *c1 = ChunkFromHandle(h1);
//     Chunk *c2 = ChunkFromHandle(h2);
//     // We can only merge chunks that are not in use.
//     HT_ASSERT_EQ(!c1->is_used() && !c2->is_used(), true);

//     // c1's prev doesn't change, still points to the same ptr, and is
//     // still not in use.

//     // Fix up neighbor pointers
//     //
//     // c1 <-> c2 <-> c3 should become
//     // c1 <-> c3

//     CUDABFCMemoryPool::ChunkHandle h3 = c2->next;
//     c1->next = h3;
//     HT_ASSERT_EQ(c2->prev, h1);
//     if (h3 != kInvalidChunkHandle) {
//         CUDABFCMemoryPool::Chunk *c3 = ChunkFromHandle(h3);
//         c3->prev = h1;
//     }

//     // Set the new size
//     c1->size += c2->size;

//     // Pick latest free time.
//     c1->freed_at_count = std::max(c1->freed_at_count, c2->freed_at_count);

//     DeleteChunk(h2);
// }

// void CUDABFCMemoryPool::DeleteChunk(ChunkHandle h) {
//     // Delete h and cleanup all state
//     Chunk *c = ChunkFromHandle(h);
//     //  VLOG(4) << "Removing: " << c->ptr;
//     _region_manager.erase(c->ptr);
//     DeallocateChunk(h);
// }

// void CUDABFCMemoryPool::InsertFreeChunkIntoBin(
//     CUDABFCMemoryPool::ChunkHandle h) {
//     Chunk *c = ChunkFromHandle(h);
//     HT_ASSERT_EQ(!c->is_used() && (c->bin_num == kInvalidBinNum), true);
//     BinNum bin_num = BinNumForSize(c->size);
//     Bin *new_bin = BinFromIndex(bin_num);
//     c->bin_num = bin_num;
//     new_bin->free_chunks.insert(h);
// }

// void CUDABFCMemoryPool::RemoveFreeChunkIterFromBin(
//     CUDABFCMemoryPool::Bin::FreeChunkSet *free_chunks,
//     const CUDABFCMemoryPool::Bin::FreeChunkSet::iterator &citer) {
//     ChunkHandle h = *citer;
//     Chunk *c = ChunkFromHandle(h);
//     HT_ASSERT_EQ(!c->is_used() && (c->bin_num != kInvalidBinNum), true);
//     free_chunks->erase(citer);
//     c->bin_num = kInvalidBinNum;
// }

// void CUDABFCMemoryPool::RemoveFreeChunkFromBin(
//     CUDABFCMemoryPool::ChunkHandle h) {
//     Chunk *c = ChunkFromHandle(h);
//     HT_ASSERT_EQ(!c->is_used() && (c->bin_num != kInvalidBinNum), true);
//     HT_ASSERT_GT(BinFromIndex(c->bin_num)->free_chunks.erase(h), 0)
//         << "Could not find chunk in bin";
//     c->bin_num = kInvalidBinNum;
// }

// void CUDABFCMemoryPool::MarkFree(CUDABFCMemoryPool::ChunkHandle h) {
//     Chunk *c = ChunkFromHandle(h);
//     HT_ASSERT_EQ(c->is_used() && (c->bin_num == kInvalidBinNum), true);

//     // Mark the chunk as no longer in use.
//     c->allocation_id = -1;
//     // // Optionally record the free time.
//     // if (timing_counter_) {
//     //     c->freed_at_count = timing_counter_->next();
//     // }
// }

// CUDABFCMemoryPool::ChunkHandle
// CUDABFCMemoryPool::TryToCoalesce(ChunkHandle h, bool ignore_freed_at) {
//     Chunk *c = ChunkFromHandle(h);
//     if ((!ignore_freed_at) && c->freed_at_count > 0)
//         return h;
//     CUDABFCMemoryPool::ChunkHandle coalesced_chunk = h;

//     // If the next chunk is free, merge it into c and delete it.
//     if (c->next != kInvalidChunkHandle
//         && !ChunkFromHandle(c->next)->is_used()) {
//         Chunk *n = ChunkFromHandle(c->next);
//         if ((n->freed_at_count == 0) || ignore_freed_at) {
//             // HT_LOG_INFO << "Merging c->next " << n->ptr << " with c " << c->ptr;
//             RemoveFreeChunkFromBin(c->next);
//             Merge(h, c->next);
//         }
//     }

//     // If the previous chunk is free, merge c into it and delete c.
//     if (c->prev != kInvalidChunkHandle
//         && !ChunkFromHandle(c->prev)->is_used()) {
//         Chunk *n = ChunkFromHandle(c->prev);
//         if ((n->freed_at_count == 0) || ignore_freed_at) {
//             // HT_LOG_INFO << "Merging c " << c->ptr << " into c->prev " << n->ptr;
//             coalesced_chunk = c->prev;
//             RemoveFreeChunkFromBin(c->prev);
//             Merge(c->prev, h);
//         }
//     }

//     return coalesced_chunk;
// }

// bool CUDABFCMemoryPool::TracksAllocationSizes() const {
//     return true;
// }

// size_t CUDABFCMemoryPool::RequestedSize(const void *ptr) const {
//     HT_ASSERT_NE(ptr, nullptr);
//     CUDABFCMemoryPool::ChunkHandle h = _region_manager.get_handle(ptr);
//     HT_ASSERT_NE(h, kInvalidChunkHandle)
//         << "Asked for requested size of pointer we never allocated: " << ptr;
//     const CUDABFCMemoryPool::Chunk *c = ChunkFromHandle(h);
//     return c->requested_size;
// }

// size_t CUDABFCMemoryPool::AllocatedSize(const void *ptr) const {
//     CUDABFCMemoryPool::ChunkHandle h = _region_manager.get_handle(ptr);
//     HT_ASSERT_NE(h, kInvalidChunkHandle)
//         << "Asked for allocated size of pointer we never allocated: " << ptr;
//     const CUDABFCMemoryPool::Chunk *c = ChunkFromHandle(h);
//     return c->size;
// }

// int64_t CUDABFCMemoryPool::AllocationId(const void *ptr) const {
//     CUDABFCMemoryPool::ChunkHandle h = _region_manager.get_handle(ptr);
//     HT_ASSERT_NE(h, kInvalidChunkHandle)
//         << "Asked for allocation id of pointer we never allocated: " << ptr;
//     const CUDABFCMemoryPool::Chunk *c = ChunkFromHandle(h);
//     return c->allocation_id;
// }

// // DataPtr CUDAMemoryPool::AllocDataSpace(size_t num_bytes) {
// //     if (num_bytes == 0) return {nullptr, 0, Device(kCUDA, _device_id)};

// //     auto alignment = get_data_alignment();
// //     if (num_bytes % alignment != 0)
// //         num_bytes = ((num_bytes / alignment) + 1) * alignment;
// //     DeviceIndex prev_id = SetDevice();
// //     void *ptr;
// //     CudaMalloc(&ptr, num_bytes);
// //     ResetDevice(prev_id);
// //     this->_allocated += num_bytes;
// //     return {ptr, num_bytes, Device(kCUDA, _device_id)};
// // }

// // void CUDAMemoryPool::FreeDataSpace(DataPtr ptr) {
// //     if (ptr.size == 0)
// //         return;

// //     DeviceIndex prev_id = SetDevice();
// //     CudaFree(ptr.ptr);
// //     ResetDevice(prev_id);
// //     this->_allocated -= ptr.size;
// // }

// DeviceIndex CUDABFCMemoryPool::SetDevice() {
//     int cur_id = -1;
//     CudaGetDevice(&cur_id);
//     if (cur_id != _device_id)
//         CudaSetDevice(_device_id);
//     return cur_id;
// }

// void CUDABFCMemoryPool::ResetDevice(DeviceIndex prev_id) {
//     if (prev_id != _device_id)
//         CudaSetDevice(prev_id);
// }

// namespace {

// static std::vector<std::shared_ptr<CUDABFCMemoryPool>> cuda_bfc_memory_pools;
// static std::once_flag cuda_bfc_memory_pool_register_flag;

// struct CUDABFCMemoryPoolRegister {
//     CUDABFCMemoryPoolRegister() {
//         std::call_once(cuda_bfc_memory_pool_register_flag, []() {
//             int32_t num_devices;
//             CudaGetDeviceCount(&num_devices);
//             std::string dev_str = std::string("Device:");
//             for (int32_t i = 0; i < num_devices; i++) {
//                 CudaSetDevice(i);
//                 size_t freeMem;
//                 size_t totalMem;
//                 cudaMemGetInfo(&freeMem, &totalMem);
//                 auto pool = std::make_shared<CUDABFCMemoryPool>(
//                     static_cast<DeviceIndex>(i),
//                     totalMem, dev_str + std::to_string(i),
//                     true, true, 0);
//                     cuda_bfc_memory_pools.push_back(pool);
//                 RegisterMemoryPool(pool);
//             }
//         });
//     }
// };

// static CUDABFCMemoryPoolRegister cuda_bfc_memory_pool_register;

// } // namespace

// } // namespace impl
// } // namespace hetu
