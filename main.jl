include("./src/jvap.jl")
using .jvap

# Example usage
word = "cross"
if is_verilogams_keyword(word)
    println("$word is a Verilog-AMS keyword in category: ", get_keyword_category(word))
else
    println("$word is not a Verilog-AMS keyword")
end