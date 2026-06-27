from agent.commands.help import help_command
from agent.commands.analyze import analyze_command
from agent.commands.architecture import architecture_command
from agent.commands.impact import impact_command


COMMANDS = {
    "help": help_command,
    "analyze": analyze_command,
    "architecture": architecture_command,
    "impact": impact_command,
}