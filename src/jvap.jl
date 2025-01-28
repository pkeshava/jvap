module jvap

export is_verilogams_keyword, get_keyword_category

"""
Dictionary of Verilog-AMS keywords categorized by their types
"""
const keywords = Dict(
    "reserved_keywords" => [
        "above", "abs", "absdelay", "absdelta", "abstol", "access", "acos", "acosh", "ac_stim", 
        "aliasparam", "always", "analog", "analysis", "and", "asin", "asinh", "assert", "assign", 
        "atan", "atan2", "atanh", "automatic", "begin", "branch", "buf", "bufif0", "bufif1", 
        "case", "casex", "casez", "ceil", "cell", "cmos", "config", "connect", "connectmodule", 
        "connectrules", "continuous", "cos", "cosh", "cross", "ddt", "ddt_nature", "ddx", 
        "deassign", "default", "defparam", "design", "disable", "discipline", "discrete", 
        "domain", "driver_update", "edge", "else", "end", "endcase", "endconfig", "endconnectrules", 
        "enddiscipline", "endfunction", "endgenerate", "endmodule", "endnature", "endparamset", 
        "endprimitive", "endspecify", "endtable", "endtask", "event", "exclude", "exp", "final_step", 
        "flicker_noise", "floor", "flow", "for", "force", "forever", "fork", "from", "function", 
        "generate", "genvar", "ground", "highz0", "highz1", "hypot", "idt", "idtmod", "idt_nature", 
        "if", "ifnone", "incdir", "include", "inf", "initial", "initial_step", "inout", "input", 
        "instance", "integer", "join", "laplace_nd", "laplace_np", "laplace_zd", "laplace_zp", 
        "large", "last_crossing", "liblist", "library", "limexp", "ln", "localparam", "log", 
        "macromodule", "max", "medium", "merged", "min", "module", "nand", "nature", "negedge", 
        "net_resolution", "nmos", "noise_table", "noise_table_log", "nor", "noshowcancelled", "not", 
        "notif0", "notif1", "or", "output", "parameter", "paramset", "pmos", "posedge", "potential", 
        "pow", "primitive", "pull0", "pull1", "pulldown", "pullup", "pulsestyle_onevent", 
        "pulsestyle_ondetect", "rcmos", "real", "realtime", "reg", "release", "repeat", "resolveto", 
        "rnmos", "rpmos", "rtran", "rtranif0", "rtranif1", "scalared", "sin", "sinh", "showcancelled", 
        "signed", "small", "specify", "sqrt", "string", "strong0", "strong1", "strong", "sum", 
        "supply0", "supply1", "table", "tan", "tanh", "task", "time", "timeprecision", "timeunit", 
        "tran", "tranif0", "tranif1", "transition", "unsigned", "use", "vectored", "wait", "wand", 
        "weak0", "weak1", "wire", "with", "wor", "xnor", "xor"
    ],
    "operators" => [
        "+", "-", "*", "/", "**", "%",  # Arithmetic
        ">", ">=", "<", "<=", "==", "!=", "===", "!==",  # Relational
        "&&", "||", "!",  # Logical
        "~", "&", "|", "^", "^~", "~^",  # Bitwise
        "&", "~&", "|", "~|", "^", "~^",  # Reduction
        "<<", ">>", "<<<", ">>>",  # Shift
        "?:",  # Conditional
        "<+"  # Contribution operator
    ],
    "system_tasks" => [
        "\$display", "\$strobe", "\$monitor", "\$stop", "\$finish", 
        "\$time", "\$realtime", "\$random", "\$fopen", "\$fclose", 
        "\$fdisplay", "\$fstrobe", "\$fmonitor"
    ],
    "compiler_directives" => [
        "`define", "`include", "`ifdef", "`ifndef", "`else", "`endif", 
        "`timescale", "`celldefine", "`endcelldefine", "`resetall", 
        "`line", "`unconnected_drive", "`nounconnected_drive", 
        "`default_nettype", "`pragma", "`begin_keywords", "`end_keywords"
    ]
)

""" 
Function to check if a word is a Verilog-AMS keyword
"""
function is_verilogams_keyword(word::String)
    for category in values(keywords)
        if word in category
            return true
        end
    end
    return false
end

"""
Function to retrieve the category of a keyword
"""
function get_keyword_category(word::String)
    for (key, category) in keywords
        if word in category
            return key
        end
    end
    return "unknown"
end

end  # module VerilogAMSKeywords