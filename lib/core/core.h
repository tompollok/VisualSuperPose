#pragma once

#include <memory>
#include <string>

#include "ppbafloc-core_export.h"

struct Image;

bool PPBAFLOC_CORE_EXPORT
sortTupleListBySecElemImg(const std::pair<std::shared_ptr<Image>, double> a,
                          const std::pair<std::shared_ptr<Image>, double> b);
bool PPBAFLOC_CORE_EXPORT
sortTupleListBySecElemPath(const std::pair<std::string, double> a,
                           const std::pair<std::string, double> b);
bool PPBAFLOC_CORE_EXPORT sortTupleListBySecElemID(
    const std::pair<int, double> a, const std::pair<int, double> b);
