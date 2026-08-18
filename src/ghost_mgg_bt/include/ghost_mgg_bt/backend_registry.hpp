#pragma once

#include <string>
#include <unordered_map>

namespace ghost_mgg_bt
{

struct BackendEndpoint
{
  std::string recover_action;
};

struct ExecutorEndpoint
{
  std::string execute_action;
};

class BackendRegistry
{
public:
  static BackendRegistry from_file(const std::string & path);

  const BackendEndpoint & backend(const std::string & name) const;
  const ExecutorEndpoint & executor(const std::string & name) const;

  void add_backend(const std::string & name, BackendEndpoint endpoint);
  void add_executor(const std::string & name, ExecutorEndpoint endpoint);

private:
  std::unordered_map<std::string, BackendEndpoint> backends_;
  std::unordered_map<std::string, ExecutorEndpoint> executors_;
};

}  // namespace ghost_mgg_bt
