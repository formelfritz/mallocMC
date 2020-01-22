/*
  mallocMC: Memory Allocator for Many Core Architectures.
  https://www.hzdr.de/crp

  Copyright 2014 Institute of Radiation Physics,
                 Helmholtz-Zentrum Dresden - Rossendorf

  Author(s):  Carlchristian Eckert - c.eckert ( at ) hzdr.de

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in
  all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
  THE SOFTWARE.
*/

#pragma once

#include <boost/mpl/int.hpp>
#include <boost/mpl/bool.hpp>

// basic files for mallocMC
#include "src/include/mallocMC/mallocMC_hostclass.hpp"

// Load all available policies for mallocMC
#include "src/include/mallocMC/CreationPolicies.hpp"
#include "src/include/mallocMC/DistributionPolicies.hpp"
#include "src/include/mallocMC/OOMPolicies.hpp"
#include "src/include/mallocMC/ReservePoolPolicies.hpp"
#include "src/include/mallocMC/AlignmentPolicies.hpp"
    

// configurate the CreationPolicy "Scatter" to modify the default behaviour
struct ScatterHeapConfig : mallocMC::CreationPolicies::Scatter<>::HeapProperties{
    //! 2MiB page can hold around 256 particle frames
    typedef boost::mpl::int_< 2 * 1024 * 1024 >  pagesize;
        /** accessblocks, regionsize and wastefactor are not conclusively
         * investigated and might be performance sensitive for multiple
         * particle species with heavily varying attributes (frame sizes)
         */
    typedef boost::mpl::int_< 4 >     accessblocks;
    typedef boost::mpl::int_< 8 >    regionsize;
    typedef boost::mpl::int_< 2 >     wastefactor;
        /** resetfreedpages is used to minimize memory fragmentation with
         * varying frame sizes
         */
    typedef boost::mpl::bool_< true > resetfreedpages;
};

// Define a new allocator and call it ScatterAllocator
// which resembles the behaviour of ScatterAlloc
typedef mallocMC::Allocator< 
  mallocMC::CreationPolicies::Scatter<ScatterHeapConfig>,
  mallocMC::DistributionPolicies::Noop,
  mallocMC::OOMPolicies::ReturnNull,
  mallocMC::ReservePoolPolicies::SimpleCudaMalloc,
  mallocMC::AlignmentPolicies::Shrink<>
  > ScatterAllocator;
