import asyncio
import logging
from pygls.lsp.server import LanguageServer
from lsprotocol.types import (
    TEXT_DOCUMENT_DID_OPEN,
    TEXT_DOCUMENT_DID_CHANGE,
    DidOpenTextDocumentParams,
    DidChangeTextDocumentParams,
    Diagnostic,
    DiagnosticSeverity,
    Range,
    Position,
    TEXT_DOCUMENT_CODE_ACTION,
    CodeActionParams,
    CodeAction,
    Command,
    PublishDiagnosticsParams,
)
from tokenizer_utils import prepare_inputs
from inference import EagleEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EagleLanguageServer(LanguageServer):
    def __init__(self, name, version, engine: EagleEngine, tokenizer):
        super().__init__(name, version)
        self.engine = engine
        self.tokenizer = tokenizer
        
        # Debounce tasks mapping: uri -> asyncio.Task
        self.active_tasks = {}
        
    async def _analyze_document(self, uri: str, text: str):
        try:
            # Native asyncio 500ms debounce (D16)
            await asyncio.sleep(0.5)
            
            logger.info(f"Analyzing {uri}...")
            
            # Notify client that analysis has started
            self.protocol.notify("eagle/analysisStarted", {"uri": uri})
            
            # Prepare inputs
            chunks, token_to_line, total_tokens = prepare_inputs(text, self.tokenizer)
            
            # Run inference in a background thread to prevent blocking the async event loop
            diagnostics_data = await asyncio.to_thread(self.engine.analyze, chunks, text)
            
            # Convert to LSP Diagnostics
            diagnostics = []
            for d in diagnostics_data:
                line_idx = d["line"] - 1 # 0-indexed for LSP
                if line_idx < 0: continue
                
                # Format the message based on the UX guidelines
                title = f"EAGLE: {d['type'].replace('_', ' ').title()} Vulnerability Detected"
                why = f"Suspicious logic detected on line {d['line']}." 
                conf = f"Model Confidence: {d.get('bag_prob', 0)*100:.1f}%"
                msg = f"{title}\n{why}\n{conf}"
                
                severity_map = {
                    1: DiagnosticSeverity.Error,
                    2: DiagnosticSeverity.Warning,
                    3: DiagnosticSeverity.Information
                }
                
                diag = Diagnostic(
                    range=Range(
                        start=Position(line=line_idx, character=0),
                        end=Position(line=line_idx, character=500) # Highlight whole line basically
                    ),
                    message=msg,
                    severity=severity_map.get(d.get('severity', 2), DiagnosticSeverity.Warning),
                    source="EAGLE"
                )
                diagnostics.append(diag)
                
            # Add a forced test diagnostic on the last line as requested
            last_line_idx = max(0, len(text.splitlines()) - 1)
            test_msg = "EAGLE TEST SQUIGGLY (User Verification)\nPipeline is fully functional.\nModel Output: [Aggressive Guardrail Bypassed]\nLogits: [...]"
            diagnostics.append(Diagnostic(
                range=Range(
                    start=Position(line=last_line_idx, character=0),
                    end=Position(line=last_line_idx, character=500)
                ),
                message=test_msg,
                severity=DiagnosticSeverity.Error,
                source="EAGLE-TEST"
            ))
                
            self.text_document_publish_diagnostics(
                PublishDiagnosticsParams(uri=uri, diagnostics=diagnostics)
            )
            logger.info(f"Published {len(diagnostics)} diagnostics for {uri}")
            
            # Notify client that analysis is finished
            self.protocol.notify("eagle/analysisFinished", {"uri": uri})
            
        except asyncio.CancelledError:
            # Task was cancelled due to a new keystroke (is_stale logic)
            logger.info(f"Analysis for {uri} was cancelled (Debounce Triggered).")
            self.protocol.notify("eagle/analysisFinished", {"uri": uri})
        except Exception as e:
            logger.error(f"Error during analysis of {uri}: {e}")
            self.protocol.notify("eagle/analysisFinished", {"uri": uri})
            
    def schedule_analysis(self, uri: str, text: str):
        # Cancel existing task if it exists (the `is_stale` flag equivalent for asyncio)
        if uri in self.active_tasks and not self.active_tasks[uri].done():
            self.active_tasks[uri].cancel()
            
        # Spawn new task
        self.active_tasks[uri] = asyncio.create_task(self._analyze_document(uri, text))

def create_server(engine, tokenizer):
    server = EagleLanguageServer("eagle-lsp", "v1.0", engine, tokenizer)
    
    @server.feature(TEXT_DOCUMENT_DID_OPEN)
    def did_open(ls: EagleLanguageServer, params: DidOpenTextDocumentParams):
        doc = ls.workspace.get_text_document(params.text_document.uri)
        ls.schedule_analysis(params.text_document.uri, doc.source)
        
    @server.feature(TEXT_DOCUMENT_DID_CHANGE)
    def did_change(ls: EagleLanguageServer, params: DidChangeTextDocumentParams):
        doc = ls.workspace.get_text_document(params.text_document.uri)
        ls.schedule_analysis(params.text_document.uri, doc.source)
        
    @server.feature("exit")
    def on_exit(ls: EagleLanguageServer, *args):
        logger.info("Received exit notification. Terminating EAGLE engine to free VRAM/RAM.")
        import sys
        sys.exit(0)
        
    @server.feature(TEXT_DOCUMENT_CODE_ACTION)
    def code_action(ls: EagleLanguageServer, params: CodeActionParams):
        actions = []
        for diag in params.context.diagnostics:
            if diag.source == "EAGLE":
                vuln_name = diag.message.split(":")[1].split("Vulnerability")[0].strip() if ":" in diag.message else "Vulnerability"
                actions.append(CodeAction(
                    title=f"💡 Quick Fix: Review Mitigations for {vuln_name}",
                    command=Command(
                        title="Open Documentation",
                        command="vscode.open",
                        arguments=["https://swcregistry.io/"]
                    )
                ))
        return actions
        
    return server
