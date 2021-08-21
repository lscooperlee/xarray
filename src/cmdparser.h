#include <algorithm>
#include <cstring>
#include <functional>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

template <typename... T>
struct Command {
    Command(
        std::vector<std::string> keywords,
        std::string description,
        std::function<void(T...)> action = [](T...) {})
        : Keywords(keywords)
        , Description(description)
        , Action(action)
    {
    }

    virtual ~Command() = default;

    int NumOfParam = sizeof...(T);
    std::vector<std::string> Keywords = {};
    std::string Description = "";
    std::function<void(T...)> Action = {};
    std::tuple<T...> Params = {};
    bool IsSpecified = false;
};

template <typename T>
bool ParseCommandOpt(const char* args, T& Commands)
{
    int argc = 0;
    int len = strlen(args);
    const char* argv[len];

    char arg_copy[len + 1];
    memset(arg_copy, 0, len + 1);
    char* p = arg_copy;

    strcpy(arg_copy, args);

    argv[argc++] = p;

    while (*++p) {
        if (*p == ' ') {
            *p = 0;
        } else if (*p != ' ' && *(p - 1) == 0) {
            argv[argc++] = p;
        }
    };

    return ParseCommandOpt(argc, argv, Commands);
}

template <typename T>
bool ParseCommandOpt(int argc, char const* const argv[], T& Commands)
{
    int index = 1;
    bool ret = true;

    auto cmdparser = [&ret, &index, argc, argv](auto&& cmd) {
        for (int i = index; i < argc; ++i) {
            auto keyword = argv[i];

            if (std::find(std::cbegin(cmd.Keywords), std::cend(cmd.Keywords), keyword) != std::cend(cmd.Keywords)) {

                ++i;

                auto paramparser = [&ret, &i, argv](auto&& param) {
                    auto cmd_param = argv[i++];

                    if (cmd_param == nullptr) {
                        ret = false;
                        return;
                    }

                    if (cmd_param[0] != '-') {
                        std::istringstream ss(cmd_param);
                        ss >> param;
                    }
                };

                std::apply([paramparser](auto&&... param) { (paramparser(param), ...); },
                    cmd.Params);

                cmd.IsSpecified = true;

                break;
            }
        }
    };

    std::apply([cmdparser](auto&&... cmd) { (cmdparser(cmd), ...); }, Commands);

    return ret;
}

//Could use Abbreviated function template in C++20
template <typename T>
void RunCommand(T& Commands)
{

    auto runeach = [](auto&& cmd) {
        if (cmd.IsSpecified) {
            std::apply(cmd.Action, cmd.Params);
        }
    };

    std::apply([runeach](auto&&... cmd) { (runeach(cmd), ...); }, Commands);
}
