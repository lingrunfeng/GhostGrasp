#include "ghost_mgg_bt/backend_registry.hpp"

#include <stdexcept>
#include <utility>

#include <yaml-cpp/yaml.h>

namespace ghost_mgg_bt
{

BackendRegistry BackendRegistry::from_file(const std::string & path)
{
  BackendRegistry registry;
  const auto root = YAML::LoadFile(path);

  if (root["backends"]) {
    for (const auto backend : root["backends"]) {
      const auto name = backend.first.as<std::string>();
      const auto recover_action = backend.second["recover_action"].as<std::string>();
      registry.add_backend(name, BackendEndpoint{recover_action});
    }
  }

  if (root["executors"]) {
    for (const auto executor : root["executors"]) {
      const auto name = executor.first.as<std::string>();
      const auto execute_action = executor.second["execute_action"].as<std::string>();
      registry.add_executor(name, ExecutorEndpoint{execute_action});
    }
  }

  return registry;
}

const BackendEndpoint & BackendRegistry::backend(const std::string & name) const
{
  return backends_.at(name);
}

const ExecutorEndpoint & BackendRegistry::executor(const std::string & name) const
{
  return executors_.at(name);
}

void BackendRegistry::add_backend(const std::string & name, BackendEndpoint endpoint)
{
  backends_[name] = std::move(endpoint);
}

void BackendRegistry::add_executor(const std::string & name, ExecutorEndpoint endpoint)
{
  executors_[name] = std::move(endpoint);
}

}  // namespace ghost_mgg_bt
