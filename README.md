# cashmere

## Online spatial search trees

Under development. Documentation may be lacking, bugs may exist.

### Motivation

Many libraries exist implementing *k*-dimensional spatial partitioning trees;
the underlying algorithms are relatively straightforward and have many
applications. However, for various reasons some features often taken for granted
in other common data structures are rarely found:

* **Mutability:** Many *k*-d tree implementations opt for a build-once,
  query-many-times design. Removing or even adding data to the structure once
  it is initialized is often impossible.
* **Balance:** In structures that *do* implement mutability, balance guarantees
  are often absent. Outer bounds for the may need to be provided, outside which
  the tree will either be pathologically imbalanced or not accept data at all.
  Regions of higher density in the data may have an unavoidably greater tree
  depth, and trees whose data is first inserted pathologically sorted in any way
  may degenerate to quadratic performance.
* **Other constraints:**
    * Most implementations accept a limited number of data types, and function
      in a limited number of dimensions.
    * Many (if not most) implementations that allow mutation require knowing
      both the inserted value *and* the associated data to change it, mandating
      a secondary data structure for some use cases.
    * Even world-class *k*-d tree implementations may suffer from seemingly
      bizarre limitations, such as an inability to handle too many items that
      coexist on a single axis-aligned plane.

This is the author's attempt to remedy all those problems, and more besides.

### Goals

* **Online mutability:** Performant and easy mutation of the values. The
  structure should be immediately available for searching in between every
  modification.
* **Performance guarantees:** Inserting data incrementally in any order should
  be reasonably fast and not result in poor search performance. No bounding box
  of the contained data should need to be known ahead of time, and no
  combination or repetition of values should result in a crash or an
  unusable/invalid structure. The structure should also not be limited in size:
  ideally there should be very few knobs to tune performance, and they should
  not be tradeoffs with maximum capacity.
* **Speed:** Ideally the data structures herein should be competitive with
  world-class search trees in terms of search performance, even when built
  incrementally. Tradeoffs between housekeeping costs during mutation and search
  performance should be customizable. As of this writing overall performance is
  circa 50% that of the best competition (`kiddo`) in a reasonably
  non-pathological benchmark, intermixing mutation and nearest-neighbor searches
  with well-distributed data. Coming revisions and rewrites are likely to close
  this gap significantly.
* **Flexibility:**
    * **of accepted types** -- The structures herein operate on traits, not
      specific predetermined numeric types. Any custom type should do, including
      even types with heterogenously typed axes. Currently the needed traits are
      defined for all primitives, `str`, `std::time` types, arrays thereof,
      references thereof, and heterogenous tuples thereof up to 6 dimensions.
      Implementing value types for whatever does not already work should be
      nearly trivial.
    * **of capabilities** -- The structures herein can be parametrized with
      *statistics*, customizable structures that can aggregate the values found
      in each part of the tree (such as bounding boxes), enhancing both the
      types of items that can be effectively searched and the ways that they can
      be searched for. This meshes with a flexible visitor-pattern query API
      enabling almost any type of query: Exact nearest searches, nearest and
      k-nearest searches tolerating errors, searches within a radius or area,
      collision and overlap searches: basic implementations will be provided
      (pending), and if it doesn't exist it can probably be created.

### Non-goals

* **Rapid serialization/deserialization:** `kiddo` + `rkyv` seem to have this
  cornered. Querying out of a cold memmapped file is a very different kind of
  goal to optimize for than performant mutability, and at this time it is
  not a priority and isn't clear whether it is possible to do both well.
* **Memory efficiency:** While the structures can be quite efficient, and will
  (as a bonus) release memory as items are removed, it does use heap allocations
  proportionate to the number of items stored and keeps pointers between them.
* **Groundbreaking algorithmic bounds:** *Balanced* *k*-d trees are generally
  known to cost `O(log^2 n)` to modify. Alternatives that perform differently
  are strangely designed with odd tradeoffs (see "divided *k*-d trees," which
  may cost `O(sqrt n)` to query) or are simply expressing their performance
  characteristics in terms of ideally distributed inputs -- which is,
  admittedly, often the case. This crate uses a variation of
  [scapegoat](https://en.wikipedia.org/wiki/Scapegoat_tree) balancing to fight
  worst-case unbalance (hence the name!). This also means **performance
  guarantees are amortized**, so while degenerate cases are avoided (by
  balancing at all) there may not be ideal bounds on latency (as rebalancing may
  occasionally be expensive, up to and including a full rebuild).
* **Ideal vectorization for each type:** At least for now, we will seek to do
  our best to lay out data in a tantalizing way, then lean on the compiler for
  the rest.
* **Great performance without LTO:** Don't forget to turn on thin-LTO!
