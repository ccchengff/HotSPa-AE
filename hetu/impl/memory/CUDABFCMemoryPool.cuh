#pragma once
/*
#include "hetu/core/memory_pool.h"
#include <string>
#include <unordered_set>
#define INF 1 << 31
#define uint64 unsigned long long

namespace hetu { namespace impl {

class CUDABFCMemoryPool final: public MemoryPool {
public:
    bool _allow_growth = true;
    // Whether the allocator will deallocate free regions to avoid OOM due to
    // memory fragmentation.
    bool _garbage_collection = false;
    // Control when a chunk should be split.
    double _fragmentation_fraction = 0;

    CUDABFCMemoryPool(DeviceIndex device_id, size_t total_memory,
                      const std::string &name, bool allow_growth,
                      bool garbage_collection, double fragmentation_fraction);

    ~CUDABFCMemoryPool();

    std::string Name(){return _name;};

    DataPtr AllocDataSpace(size_t num_bytes);

    void FreeDataSpace(DataPtr ptr);

    bool TracksAllocationSizes() const;

    size_t RequestedSize(const void* ptr) const;

    size_t AllocatedSize(const void* ptr) const;

    int64_t AllocationId(const void* ptr) const;

    inline Device device() {
        return {kCUDA, _device_id};
    }

    inline size_t get_data_alignment() const noexcept {
        return 256;
    }

private:
    struct Bin;

    DeviceIndex SetDevice();
    void ResetDevice(DeviceIndex prev_id);

    void *AllocateRawInternal(size_t num_bytes);

    // void DeallocateRaw(void *ptr);
    // void FreeDataSpace(DataPtr ptr);
    void DeallocateRawInternal(void *ptr);

    // A ChunkHandle is an index into the chunks_ vector in BFCAllocator
    // kInvalidChunkHandle means an invalid chunk
    typedef size_t ChunkHandle;
    static constexpr ChunkHandle kInvalidChunkHandle = 1 << 31;

    typedef int BinNum;
    static constexpr int kInvalidBinNum = -1;
    // The following means that the largest bin'd chunk size is 256 << 21 =
    // 512MB.
    static constexpr int kNumBins = 21;

    // A single chunk represents a piece of continous memory.
    struct Chunk {
        size_t size = 0; // size for this chunk, s.t., size % 256 == 0.
        size_t requested_size = 0; // used size for  this chunk
        int64_t allocation_id =
            -1;              // flag for current status, e.g., -1 for available
        void *ptr = nullptr; // the pointer to the chunk memory
        ChunkHandle prev =
            kInvalidChunkHandle; // pointer/index for the previous chunk
        ChunkHandle next =
            kInvalidChunkHandle;         // pointer/index for the next chunk
        BinNum bin_num = kInvalidBinNum; // Index of the corresponding Bin
        uint64 freed_at_count = 0;
        bool is_used() const {
            return allocation_id != -1;
        }
    };

    // A Bin is a collection of similar-sized free chunks.
    // Allocated chunks are never in a Bin.
    struct Bin {
        // All chunks in this bin have >= bin_size memory.
        size_t bin_size = 0;

        class ChunkComparator {
        public:
            explicit ChunkComparator(CUDABFCMemoryPool *allocator) :
                allocator_(allocator) {
            }
            // Sort first by size and then use pointer address as a tie breaker.
            bool operator()(const ChunkHandle ha, const ChunkHandle hb) const {
                const Chunk *a = allocator_->ChunkFromHandle(ha);
                const Chunk *b = allocator_->ChunkFromHandle(hb);
                if (a->size != b->size) {
                    return a->size < b->size;
                }
                return a->ptr < b->ptr;
            }

        private:
            CUDABFCMemoryPool *allocator_; // The parent allocator
        };

        typedef std::set<ChunkHandle, ChunkComparator> FreeChunkSet;
        // List of free chunks within the bin, sorted by chunk size.
        // Chunk * not owned.
        FreeChunkSet free_chunks;
        Bin(CUDABFCMemoryPool *allocator, size_t bs) :
            bin_size(bs), free_chunks(ChunkComparator(allocator)) {
        }
    };

    static constexpr size_t kMinAllocationBits = 8;
    static constexpr size_t kMinAllocationSize = 1 << kMinAllocationBits;

    class AllocationRegion {
    public:
        AllocationRegion(void *ptr, size_t memory_size) :
            ptr_(ptr), memory_size_(memory_size),
            end_ptr_(
                static_cast<void *>(static_cast<char *>(ptr_) + memory_size_)) {
            HT_ASSERT_EQ(0, memory_size % kMinAllocationSize);
            const size_t n_handles =
                (memory_size + kMinAllocationSize - 1) / kMinAllocationSize;
            handles_.resize(n_handles, kInvalidChunkHandle);
        }

        AllocationRegion() = default;
        AllocationRegion(AllocationRegion &&other) {
            Swap(&other);
        }
        AllocationRegion &operator=(AllocationRegion &&other) {
            Swap(&other);
            return *this;
        }

        void *ptr() const {
            return ptr_;
        }
        void *end_ptr() const {
            return end_ptr_;
        }
        size_t memory_size() const {
            return memory_size_;
        }
        void extend(size_t size) {
            memory_size_ += size;
            HT_ASSERT_EQ(0, memory_size_ % kMinAllocationSize);

            end_ptr_ =
                static_cast<void *>(static_cast<char *>(end_ptr_) + size);
            const size_t n_handles =
                (memory_size_ + kMinAllocationSize - 1) / kMinAllocationSize;
            handles_.resize(n_handles, kInvalidChunkHandle);
        }
        ChunkHandle get_handle(const void *p) const {
            return handles_[IndexFor(p)];
        }
        void set_handle(const void *p, ChunkHandle h) {
            handles_[IndexFor(p)] = h;
        }
        void erase(const void *p) {
            set_handle(p, kInvalidChunkHandle);
        }

    private:
        void Swap(AllocationRegion *other) {
            std::swap(ptr_, other->ptr_);
            std::swap(memory_size_, other->memory_size_);
            std::swap(end_ptr_, other->end_ptr_);
            std::swap(handles_, other->handles_);
        }

        size_t IndexFor(const void *p) const {
            std::uintptr_t p_int = reinterpret_cast<std::uintptr_t>(p);
            std::uintptr_t base_int = reinterpret_cast<std::uintptr_t>(ptr_);
            HT_ASSERT_GE(p_int, base_int);
            HT_ASSERT_LT(p_int, base_int + memory_size_);
            return static_cast<size_t>(
                ((p_int - base_int) >> kMinAllocationBits));
        }

        // Metadata about the allocation region.
        void *ptr_ = nullptr;
        size_t memory_size_ = 0;
        void *end_ptr_ = nullptr;

        // Array of size "memory_size / kMinAllocationSize".  It is
        // indexed by (p-base) / kMinAllocationSize, contains ChunkHandle
        // for the memory allocation represented by "p"
        std::vector<ChunkHandle> handles_;
    };

    class RegionManager {
    public:
        RegionManager() {
        }
        ~RegionManager() {
        }

        void AddAllocationRegion(void *ptr, size_t memory_size) {
            // Insert sorted by end_ptr.
            auto entry = std::upper_bound(regions_.begin(), regions_.end(), ptr,
                                          &Comparator);
            regions_.insert(entry, AllocationRegion(ptr, memory_size));
        }

        // Adds an alloation region for the given ptr and size, potentially
        // extending a region if ptr matches the end_ptr of an existing region.
        // If a region is extended, returns a pointer to the extended region so
        // that the BFC allocator can reason about chunkification.
        AllocationRegion *AddOrExtendAllocationRegion(void *ptr,
                                                      size_t memory_size) {
            // Insert sorted by end_ptr.
            auto entry = std::upper_bound(regions_.begin(), regions_.end(), ptr,
                                          &Comparator);
            // Check if can be coalesced with preceding region.
            if (entry != regions_.begin()) {
                auto preceding_region = entry - 1;
                if (preceding_region->end_ptr() == ptr) {                    
                        // HT_LOG_INFO << "Extending region "
                        //           << preceding_region->ptr() << " of "
                        //           << strings::HumanReadableNumBytes(
                        //                  preceding_region->memory_size())
                        //           << "  by "
                        //           << strings::HumanReadableNumBytes(memory_size)
                        //           << " bytes";
                    preceding_region->extend(memory_size);
                    return &*preceding_region;
                }
            }
            regions_.insert(entry, AllocationRegion(ptr, memory_size));
            return nullptr;
        }

        std::vector<AllocationRegion>::iterator
        RemoveAllocationRegion(std::vector<AllocationRegion>::iterator it) {
            return regions_.erase(it);
        }

        ChunkHandle get_handle(const void *p) const {
            return RegionFor(p)->get_handle(p);
        }

        void set_handle(const void *p, ChunkHandle h) {
            return MutableRegionFor(p)->set_handle(p, h);
        }
        void erase(const void *p) {
            return MutableRegionFor(p)->erase(p);
        }

        const std::vector<AllocationRegion> &regions() const {
            return regions_;
        }

    private:
        static bool Comparator(const void *ptr, const AllocationRegion &other) {
            return ptr < other.end_ptr();
        }

        AllocationRegion *MutableRegionFor(const void *p) {
            return const_cast<AllocationRegion *>(RegionFor(p));
        }

        const AllocationRegion *RegionFor(const void *p) const {
            auto entry = std::upper_bound(regions_.begin(), regions_.end(), p,
                                          &Comparator);

            if (entry != regions_.end()) {
                return &(*entry);
            }

            HT_LOG_INFO << "Could not find Region for " << p;
            return nullptr;
        }

    private:
        std::vector<AllocationRegion> regions_;
    };
    inline int Log2FloorNonZero(uint64 n) {
        int r = 0;
        while (n > 0) {
          r++;
          n >>= 1;
        }
        return r - 1;
    }

    static size_t RoundedBytes(size_t bytes);

    bool Extend(size_t rounded_bytes);

    bool DeallocateFreeRegions(size_t rounded_bytes);

    void DeallocateRegions(const std::unordered_set<void *> &region_ptrs);

    void *FindChunkPtr(BinNum bin_num, size_t rounded_bytes, size_t num_bytes);

    void SplitChunk(ChunkHandle h, size_t num_bytes);

    void Merge(ChunkHandle h, ChunkHandle h2);

    void InsertFreeChunkIntoBin(ChunkHandle h);

    void RemoveFreeChunkIterFromBin(Bin::FreeChunkSet *free_chunks,
                                    const Bin::FreeChunkSet::iterator &c);

    void RemoveFreeChunkFromBin(ChunkHandle h);
    
    ChunkHandle TryToCoalesce(ChunkHandle h, bool ignore_freed_at);

    void MarkFree(ChunkHandle h);
    
    void DeleteChunk(ChunkHandle h);

    ChunkHandle AllocateChunk();

    void DeallocateChunk(ChunkHandle h);

    Chunk *ChunkFromHandle(ChunkHandle h);
    const Chunk *ChunkFromHandle(ChunkHandle h) const;

    // Map from bin size to Bin
    Bin *BinFromIndex(BinNum index) {
        return reinterpret_cast<Bin *>(&(bins_space_[index * sizeof(Bin)]));
    }
    size_t BinNumToSize(BinNum index) {
        return static_cast<size_t>(256) << index;
    }
    BinNum BinNumForSize(size_t bytes) {
        uint64 v = std::max<size_t>(bytes, 256) >> kMinAllocationBits;
        int b = std::min(kNumBins - 1, Log2FloorNonZero(v));
        return b;
    }
    Bin *BinForSize(size_t bytes) {
        return BinFromIndex(BinNumForSize(bytes));
    }

    char bins_space_[sizeof(Bin) * kNumBins];

    // Structures mutable after construction
    // mutable mutex _lock;
    std::string _name;
    RegionManager _region_manager;
    std::vector<Chunk> _chunks;
    ChunkHandle _free_chunks_list;
    int64_t next_allocation_id_;
    size_t _total_memory;
    const DeviceIndex _device_id;
    size_t _allocated = 0;
    
    // The size of the current region allocation.
    size_t curr_region_allocation_bytes_;
    // The total number of allocated bytes by the allocator.
    size_t total_region_allocated_bytes_ = 0;
    // Structures immutable after construction
    size_t memory_limit_ = 0;

};

}} // namespace hetu::impl
*/