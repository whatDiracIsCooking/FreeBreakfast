module;

#include <bit>
#include <taskflow/taskflow.hpp>

export module free_breakfast.third_party.taskflow;

export namespace free_breakfast::tf {

using ::tf::Executor;
using ::tf::FlowBuilder;
using ::tf::Subflow;
using ::tf::Task;
using ::tf::Taskflow;

} // namespace free_breakfast::tf
