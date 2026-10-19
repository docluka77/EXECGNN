import copy,json,threading,unittest
from http.server import BaseHTTPRequestHandler,HTTPServer
import numpy as np
import torch
from examples.make_tiny import record
from execglm.budget import resolve_budget
from execglm.data import collate
from execglm.model import ExecCore,loss_fn
from synth.interp import ev
from synth import gen_llm


class CoreTests(unittest.TestCase):
    def test_interpreter_operand_order_and_unknowns(self):
        self.assertEqual(ev('Sub',[(7,0),(2,0)],[8,8],8),(5,0))
        self.assertEqual(ev('Sub',[(2,0),(7,0)],[8,8],8),(251,0))
        self.assertEqual(ev('Cond',[(0,1),(5,0),(5,0)],[1,8,8],8),(5,0))
        self.assertEqual(ev('BitAnd',[(0,0),(0,255)],[8,8],8),(0,0))

    def test_execution_and_no_internal_input_leakage(self):
        torch.set_num_threads(2);torch.manual_seed(0)
        r=record('sample',3);batch=collate([('sample',0,'sample_trace0')],{'sample':r})
        model=ExecCore(d=16,hidden=32).eval()
        with torch.no_grad():
            first=model(batch,6);changed=copy.deepcopy(batch)
            changed['values'][changed['role']>=2]=123
            second=model(changed,6)
        for a,b in zip(first,second):self.assertTrue(torch.equal(a,b))
        model.train();out=model(batch,6)
        loss,_=loss_fn(out,batch,6,dict(bits=1,nz=1,ch=1,tog=1,br=1))
        self.assertTrue(torch.isfinite(loss));loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_exact_update_budget(self):
        b=resolve_budget(5,2,1,max_steps=4)
        self.assertEqual((b.epochs,b.updates,b.batches_per_epoch),(2,4,3))

    def test_author_http_contract_without_external_service(self):
        captured=[]
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_POST(self):
                captured.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
                result=json.dumps({'choices':[{'message':{'content':'```verilog\nmodule tiny; endmodule\n```'}}]}).encode()
                self.send_response(200);self.send_header('Content-Type','application/json');self.end_headers();self.wfile.write(result)
        server=HTTPServer(('127.0.0.1',0),Handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            text=gen_llm.ask(f'http://127.0.0.1:{server.server_port}/v1/chat/completions','test-model','a counter',60,.8,7)
            self.assertIn('module tiny',str(text));self.assertEqual(captured[0]['model'],'test-model')
            self.assertEqual(captured[0]['temperature'],.8)
        finally:server.shutdown();server.server_close();thread.join()

if __name__=='__main__':unittest.main()
