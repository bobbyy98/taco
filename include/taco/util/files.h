#ifndef TACO_UTIL_FILES_H
#define TACO_UTIL_FILES_H

#include <fstream>
#include <string>

namespace taco { namespace util {

  std::string sanitizePath(std::string path);

  void openStream(std::fstream &stream, std::string path, std::fstream::openmode mode);

}} // namespace taco::util
#endif
