#include "core.h"

#include "import/colmapimporter.h"
#include "types/image.h"

bool sortTupleListBySecElemImg(
    const std::pair<std::shared_ptr<Image>, double> a,
    const std::pair<std::shared_ptr<Image>, double> b) {
  return a.second > b.second;
}

bool sortTupleListBySecElemPath(const std::pair<std::string, double> a,
                                const std::pair<std::string, double> b) {
  return a.second > b.second;
}

bool sortTupleListBySecElemID(const std::pair<int, double> a,
                              const std::pair<int, double> b) {
  return a.second > b.second;
}
