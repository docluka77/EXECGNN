module counter(input clk, input rst, input en, output reg [7:0] q);
always @(posedge clk) begin
    if (rst) q <= 8'd0;
    else if (en) q <= q + 8'd1;
end
endmodule
