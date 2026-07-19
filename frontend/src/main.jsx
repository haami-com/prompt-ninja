import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Badge,
  Box,
  Button,
  Container,
  Divider,
  Flex,
  FormControl,
  FormLabel,
  Heading,
  HStack,
  Icon,
  Input,
  Select,
  SimpleGrid,
  Stack,
  Text,
  Textarea,
  ChakraProvider,
} from "@chakra-ui/react";
import { FiArrowUpRight, FiCheck, FiChevronDown, FiCopy, FiCpu, FiDownload, FiFileText, FiLoader, FiMessageCircle, FiRefreshCw, FiStar, FiUploadCloud } from "react-icons/fi";
import { theme } from "./theme";
import "./styles.css";

// 127.0.0.1 avoids resolving `localhost` to a different IPv6/container listener.
const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";
const MODEL_OPTIONS = [
  ["gpt-5.6-sol", "GPT-5.6 Sol"],
  ["gpt-5.6-terra", "GPT-5.6 Terra"],
  ["gpt-5.6-luna", "GPT-5.6 Luna"],
  ["gpt-5.5", "GPT-5.5"],
  ["gpt-5.4", "GPT-5.4"],
  ["gpt-5.4-mini", "GPT-5.4 mini"],
  ["gpt-5.4-nano", "GPT-5.4 nano"],
];
const modelLabel = (model) => MODEL_OPTIONS.find(([value]) => value === model)?.[1] || model;

function IconBox({ children, ...props }) {
  return <Box display="flex" alignItems="center" justifyContent="center" flexShrink={0} {...props}>{children}</Box>;
}

const starter = {
  outcome: "",
  context: "",
  source_text: "",
  expected_output: "",
  constraints: "",
};

function App() {
  const [brief, setBrief] = useState(starter);
  const [files, setFiles] = useState([]);
  const [events, setEvents] = useState([]);
  const [result, setResult] = useState(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState("");
  const [copied, setCopied] = useState(false);
  const [downloading, setDownloading] = useState(false);
  const [testModel, setTestModel] = useState("gpt-5.6-terra");
  const [promptTest, setPromptTest] = useState(null);
  const [testingPrompt, setTestingPrompt] = useState(false);
  const [testError, setTestError] = useState("");
  const [models, setModels] = useState({ creators: ["gpt-5.6-sol", "gpt-5.6-luna", "gpt-5.5"], judge: "gpt-5.6-terra" });
  const [promptConfig, setPromptConfig] = useState({ creators: ["", "", ""], judge: "" });

  useEffect(() => {
    fetch(`${API_URL}/api/prompts`)
      .then((response) => response.ok ? response.json() : null)
      .then((defaults) => defaults && setPromptConfig(defaults))
      .catch(() => {});
  }, []);

  const update = (key) => (event) => setBrief((current) => ({ ...current, [key]: event.target.value }));
  const completed = useMemo(() => new Set(events.filter((event) => event.status === "complete").map((event) => event.stage)), [events]);

  async function generate() {
    setRunning(true); setError(""); setResult(null); setEvents([]); setPromptTest(null); setTestError("");
    const body = new FormData();
    Object.entries(brief).forEach(([key, value]) => body.append(key, value));
    body.append("creator_models", JSON.stringify(models.creators));
    body.append("judge_model", models.judge);
    body.append("creator_prompts", JSON.stringify(promptConfig.creators));
    body.append("judge_prompt", promptConfig.judge);
    files.forEach((file) => body.append("files", file));
    try {
      const response = await fetch(`${API_URL}/api/generate`, { method: "POST", body });
      if (!response.ok) throw new Error((await response.json()).detail || "The council could not start.");
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      const consume = (line) => {
        if (!line.trim()) return;
        const item = JSON.parse(line);
        if (item.type === "agent") setEvents((current) => [...current, item.data]);
        if (item.type === "result") setResult(item.data);
        if (item.type === "error") throw new Error(item.message || "The council failed while running.");
      };
      while (true) {
        const { value, done } = await reader.read();
        buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
        const lines = buffer.split(/\r?\n/);
        buffer = lines.pop() || "";
        lines.forEach(consume);
        if (done) break;
      }
      if (buffer.trim()) consume(buffer);
    } catch (runError) {
      setError(runError.message || "Something went wrong.");
    } finally { setRunning(false); }
  }

  async function copyPrompt() {
    if (!result?.final_prompt) return;
    await navigator.clipboard.writeText(result.final_prompt);
    setCopied(true); setTimeout(() => setCopied(false), 1600);
  }

  async function downloadPrompt() {
    if (!result?.final_prompt) return;
    setDownloading(true); setError("");
    try {
      const response = await fetch(`${API_URL}/api/export-prompt`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ final_prompt: result.final_prompt, goal: brief.outcome, model: models.judge }),
      });
      if (!response.ok) throw new Error((await response.json()).detail || "The prompt could not be exported.");
      const disposition = response.headers.get("Content-Disposition") || "";
      const filename = disposition.match(/filename="?([^";]+)"?/)?.[1] || "generated-prompt.prompt.toml";
      const url = URL.createObjectURL(await response.blob());
      const link = document.createElement("a");
      link.href = url; link.download = filename; link.click(); URL.revokeObjectURL(url);
    } catch (downloadError) {
      setError(downloadError.message || "The prompt could not be exported.");
    } finally { setDownloading(false); }
  }

  async function runGeneratedTest() {
    if (!result?.final_prompt) return;
    setTestingPrompt(true); setTestError(""); setPromptTest(null);
    try {
      const response = await fetch(`${API_URL}/api/test-generated`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ final_prompt: result.final_prompt, goal: brief.outcome, context: brief.context, expected_output: brief.expected_output, model: testModel, judge_model: models.judge }),
      });
      if (!response.ok) throw new Error((await response.json()).detail || "The prompt test could not run.");
      setPromptTest(await response.json());
    } catch (testRunError) {
      setTestError(testRunError.message || "The prompt test could not run.");
    } finally { setTestingPrompt(false); }
  }

  return (
    <Box className="app-shell">
      <Container maxW="1280px" py={{ base: 5, md: 8 }}>
        <Flex justify="space-between" align="center" mb={{ base: 10, md: 16 }}>
          <HStack spacing={3}><IconBox boxSize="34px" borderRadius="12px" bg="ink" color="white"><Icon as={FiStar} /></IconBox><Text fontWeight="700" letterSpacing="-0.02em">prompt council</Text></HStack>
          <HStack spacing={3} color="gray.500" fontSize="sm"><Text>Private workspace</Text><Box w="6px" h="6px" bg="sage.500" borderRadius="full" /></HStack>
        </Flex>

        <SimpleGrid columns={{ base: 1, lg: 2 }} gap={{ base: 10, lg: 20 }} alignItems="start">
          <Stack spacing={8}>
            <Box>
              <Heading mt={4} fontSize={{ base: "3xl", md: "5xl" }} lineHeight="1.04" letterSpacing="-0.055em">You describe<br />the <Box as="span" color="sage.500">expectation.</Box></Heading>
              <Text mt={5} color="gray.600" maxW="500px" fontSize="lg" lineHeight="1.6">The LLMs create the prompt. Share what you want to happen, and let them work out the instructions.</Text>
            </Box>
            <Stack spacing={5}>
              <FormControl isRequired><FormLabel fontSize="sm" fontWeight="700">What do you want to achieve?</FormLabel><Textarea value={brief.outcome} onChange={update("outcome")} placeholder="Example: Extract action items from project update documents" rows={3} bg="white" borderColor="blackAlpha.200" /></FormControl>
              <FormControl><FormLabel fontSize="sm" fontWeight="700">Where will this prompt live?</FormLabel><Input value={brief.context} onChange={update("context")} placeholder="Example: Inside an operations app used by team leads" bg="white" borderColor="blackAlpha.200" /></FormControl>
              <FormControl><FormLabel fontSize="sm" fontWeight="700">What should the result look like?</FormLabel><Textarea value={brief.expected_output} onChange={update("expected_output")} placeholder="Example: Return 3–5 bullets. Start each with the owner, action, and due date when available." rows={4} bg="white" borderColor="blackAlpha.200" /></FormControl>
              <FormControl><FormLabel fontSize="sm" fontWeight="700">Source material <Text as="span" fontWeight="400" color="gray.500">(optional)</Text></FormLabel><Textarea value={brief.source_text} onChange={update("source_text")} placeholder="Example: Paste a sample document or input here" rows={4} bg="white" borderColor="blackAlpha.200" /></FormControl>
              <Flex className="upload-zone" as="label" htmlFor="files" cursor="pointer" align="center" gap={3}><Icon as={FiUploadCloud} color="sage.500" boxSize={5} /><Box><Text fontSize="sm" fontWeight="700">Add reference files</Text><Text fontSize="xs" color="gray.500">PDF, DOCX, PNG or JPG · up to 3 files</Text></Box><Input id="files" type="file" accept=".pdf,.docx,.png,.jpg,.jpeg" multiple display="none" onChange={(event) => setFiles(Array.from(event.target.files || []))} /></Flex>
              {files.length > 0 && <HStack color="gray.600" fontSize="sm"><Icon as={FiFileText} /><Text>{files.map((file) => file.name).join(", ")}</Text></HStack>}
              <FormControl><FormLabel fontSize="sm" fontWeight="700">Guardrails <Text as="span" fontWeight="400" color="gray.500">(optional)</Text></FormLabel><Input value={brief.constraints} onChange={update("constraints")} placeholder="Example: Do not infer missing owners or dates" bg="white" borderColor="blackAlpha.200" /></FormControl>
              <Box bg="white" border="1px solid" borderColor="blackAlpha.100" borderRadius="16px" p={5}>
                <HStack spacing={3} mb={4}><Icon as={FiCpu} color="sage.500" /><Box><Text fontSize="sm" fontWeight="700">Choose the council models</Text><Text fontSize="xs" color="gray.500">Three creators propose prompts. One judge compares them.</Text></Box></HStack>
                <SimpleGrid columns={{ base: 1, md: 2 }} gap={3}>
                  {models.creators.map((model, index) => <FormControl key={`creator-${index}`}><FormLabel fontSize="xs" color="gray.500">Creator {index + 1}</FormLabel><Select value={model} onChange={(event) => setModels((current) => ({ ...current, creators: current.creators.map((item, itemIndex) => itemIndex === index ? event.target.value : item) }))} bg="gray.50" borderColor="blackAlpha.100" size="sm">{MODEL_OPTIONS.map(([value, label]) => <option key={value} value={value}>{label}</option>)}</Select></FormControl>)}
                  <FormControl><FormLabel fontSize="xs" color="gray.500">Judge · default {modelLabel("gpt-5.6-terra")}</FormLabel><Select value={models.judge} onChange={(event) => setModels((current) => ({ ...current, judge: event.target.value }))} bg="gray.50" borderColor="blackAlpha.100" size="sm">{MODEL_OPTIONS.map(([value, label]) => <option key={value} value={value}>{label}</option>)}</Select></FormControl>
                </SimpleGrid>
                <Box as="details" mt={5} pt={4} borderTop="1px solid" borderColor="blackAlpha.100"><Box as="summary" cursor="pointer" fontSize="sm" fontWeight="700">Edit agent prompts</Box><Text mt={2} fontSize="xs" color="gray.500">These instructions are sent to the selected models. The brief and creator outputs are added automatically.</Text><Stack spacing={3} mt={4}>{promptConfig.creators.map((prompt, index) => <FormControl key={`prompt-${index}`}><FormLabel fontSize="xs" color="gray.500">Creator {index + 1} instruction</FormLabel><Textarea value={prompt} onChange={(event) => setPromptConfig((current) => ({ ...current, creators: current.creators.map((item, itemIndex) => itemIndex === index ? event.target.value : item) }))} rows={3} bg="gray.50" borderColor="blackAlpha.100" fontSize="sm" /></FormControl>)}<FormControl><FormLabel fontSize="xs" color="gray.500">Judge instruction</FormLabel><Textarea value={promptConfig.judge} onChange={(event) => setPromptConfig((current) => ({ ...current, judge: event.target.value }))} rows={5} bg="gray.50" borderColor="blackAlpha.100" fontSize="sm" /></FormControl></Stack></Box>
              </Box>
              <Button onClick={generate} isLoading={running} loadingText="Council is thinking" size="lg" bg="ink" color="white" _hover={{ bg: "sage.700" }} rightIcon={<FiArrowUpRight />} borderRadius="14px" py={7}>Convene the council</Button>
              {error && <Text color="red.600" fontSize="sm">{error}</Text>}
            </Stack>
          </Stack>

          <Stack className="council-panel" spacing={5}>
            <Flex justify="space-between" align="center"><Box><Text fontSize="xs" color="gray.500" fontWeight="700" letterSpacing="0.1em">THE COUNCIL</Text><Heading fontSize="2xl" letterSpacing="-0.03em" mt={1}>From rough idea to ready-to-use</Heading></Box><Badge colorScheme={running ? "orange" : result ? "green" : "gray"} borderRadius="full" px={3}>{running ? "In session" : result ? "Complete" : "Waiting"}</Badge></Flex>
            <Box bg="white" border="1px solid" borderColor="blackAlpha.100" borderRadius="20px" p={{ base: 5, md: 7 }} boxShadow="0 20px 60px rgba(32, 49, 39, .07)">
              <Stack spacing={0}>
                {[{ stage: "requirements", icon: FiMessageCircle, label: "Requirements analyst", detail: "Mapping the expectation and edge cases" }, { stage: "creator_1", icon: FiStar, label: `Creator 1 · ${modelLabel(models.creators[0])}`, detail: "Writing a first proposal" }, { stage: "creator_2", icon: FiStar, label: `Creator 2 · ${modelLabel(models.creators[1])}`, detail: "Trying another approach" }, { stage: "creator_3", icon: FiStar, label: `Creator 3 · ${modelLabel(models.creators[2])}`, detail: "Stress-testing the instructions" }, { stage: "synthesis", icon: FiArrowUpRight, label: `Judge · ${modelLabel(models.judge)}`, detail: "Combining the useful ideas into one prompt" }].map((agent, index, stages) => {
                  const event = events.find((item) => item.stage === agent.stage && item.status === "complete"); const active = running && !event && (index === 0 || completed.has(stages[index - 1].stage));
                  return <Box key={agent.stage} className={`agent-row ${event ? "is-done" : active ? "is-active" : ""}`}><Flex gap={4} align="start"><IconBox boxSize="34px" borderRadius="12px" bg={event ? "sage.100" : active ? "coral.50" : "gray.100"} color={event ? "sage.700" : active ? "coral.500" : "gray.400"}><Icon as={active ? FiLoader : event ? FiCheck : agent.icon} /></IconBox><Box flex="1"><Flex justify="space-between" gap={3}><Text fontWeight="700" fontSize="sm">{agent.label}</Text>{event && <Text fontSize="xs" color="sage.700">done</Text>}</Flex><Text fontSize="sm" color="gray.500" mt={1}>{event?.summary || agent.detail}</Text></Box></Flex>{index < stages.length - 1 && <Box className="agent-line" />}</Box>;
                })}
              </Stack>
              {result && <Box mt={7} pt={7} borderTop="1px solid" borderColor="blackAlpha.100">
                <Text fontSize="xs" fontWeight="700" letterSpacing="0.1em" color="sage.700" mb={3}>CREATOR OUTPUTS</Text>
                <Stack spacing={3} mb={7}>{(result.creators || []).map((creator, index) => <Box key={`${creator.model}-${index}`} bg="gray.50" borderRadius="14px" p={4}><Flex justify="space-between" align="center" mb={2}><Text fontSize="sm" fontWeight="700">Creator {index + 1}</Text><Badge colorScheme="gray" fontSize="xs">{modelLabel(creator.model)}</Badge></Flex><Text fontSize="sm" color="gray.700" whiteSpace="pre-wrap" lineHeight="1.6">{creator.draft}</Text><Text mt={2} fontSize="xs" color="gray.500">{creator.rationale}</Text></Box>)}</Stack>
                <Text fontSize="xs" fontWeight="700" letterSpacing="0.1em" color="sage.700" mb={3}>PROMPT TRACE</Text>
                <Stack spacing={2} mb={7} className="prompt-trace">
                  {(result.prompt_trace?.creators || []).map((trace) => <Box as="details" key={`trace-${trace.slot}`} bg="gray.50" borderRadius="12px" px={4} py={3}><Box as="summary" cursor="pointer" fontSize="sm" fontWeight="700">Creator {trace.slot} · {modelLabel(trace.model)}</Box><Text mt={3} fontSize="xs" fontWeight="700" color="gray.500">SYSTEM PROMPT</Text><Text mt={1} fontSize="xs" whiteSpace="pre-wrap" fontFamily="mono" color="gray.700">{trace.system_prompt}</Text><Text mt={3} fontSize="xs" fontWeight="700" color="gray.500">INPUT CONTEXT</Text><Text mt={1} fontSize="xs" whiteSpace="pre-wrap" fontFamily="mono" color="gray.700" maxH="220px" overflowY="auto">{trace.input_context}</Text></Box>)}
                  {result.prompt_trace?.judge && <Box as="details" bg="coral.50" borderRadius="12px" px={4} py={3}><Box as="summary" cursor="pointer" fontSize="sm" fontWeight="700">Judge · {modelLabel(result.prompt_trace.judge.model)}</Box><Text mt={3} fontSize="xs" fontWeight="700" color="coral.500">SYSTEM PROMPT</Text><Text mt={1} fontSize="xs" whiteSpace="pre-wrap" fontFamily="mono" color="gray.700">{result.prompt_trace.judge.system_prompt}</Text><Text mt={3} fontSize="xs" fontWeight="700" color="coral.500">INPUT CONTEXT</Text><Text mt={1} fontSize="xs" whiteSpace="pre-wrap" fontFamily="mono" color="gray.700" maxH="260px" overflowY="auto">{result.prompt_trace.judge.input_context}</Text></Box>}
                </Stack>
                <Box bg="coral.50" border="1px solid" borderColor="coral.200" borderRadius="14px" p={4} mb={7}><Text fontSize="xs" fontWeight="700" letterSpacing="0.1em" color="coral.500" mb={2}>JUDGE DECISION SUMMARY</Text><Text fontSize="sm" color="gray.700" lineHeight="1.6">{result.judge_summary || "Combined the useful ideas from all three creator proposals into one prompt."}</Text></Box>
                <Flex justify="space-between" align="center" mb={3} gap={2}><Text fontSize="xs" fontWeight="700" letterSpacing="0.1em" color="sage.700">FINAL PROMPT</Text><HStack spacing={1}><Button size="sm" variant="ghost" leftIcon={copied ? <FiCheck /> : <FiCopy />} onClick={copyPrompt}>{copied ? "Copied" : "Copy"}</Button><Button size="sm" variant="ghost" leftIcon={<FiDownload />} onClick={downloadPrompt} isLoading={downloading}>Download TOML</Button></HStack></Flex><Textarea value={result.final_prompt} onChange={(event) => { setResult({ ...result, final_prompt: event.target.value }); setPromptTest(null); }} minH="260px" resize="vertical" bg="sage.50" borderColor="sage.100" fontSize="sm" lineHeight="1.7" />
                <Box mt={5} pt={5} borderTop="1px solid" borderColor="blackAlpha.100">
                  <Text fontSize="xs" fontWeight="700" letterSpacing="0.1em" color="sage.700">SELF-TEST THE GENERATED PROMPT</Text>
                  <Text mt={2} fontSize="sm" color="gray.600">Prompt Ninja creates a realistic fixture, runs this prompt with the selected model, then asks the judge whether the response meets the expectation.</Text>
                  <SimpleGrid columns={{ base: 1, md: 2 }} gap={3} mt={4}>
                    <FormControl><FormLabel fontSize="xs" color="gray.500">Runner model</FormLabel><Select value={testModel} onChange={(event) => setTestModel(event.target.value)} size="sm" bg="gray.50">{MODEL_OPTIONS.map(([value, label]) => <option key={value} value={value}>{label}</option>)}</Select></FormControl>
                    <FormControl><FormLabel fontSize="xs" color="gray.500">Judge model</FormLabel><Select value={models.judge} onChange={(event) => setModels((current) => ({ ...current, judge: event.target.value }))} size="sm" bg="gray.50">{MODEL_OPTIONS.map(([value, label]) => <option key={value} value={value}>{label}</option>)}</Select></FormControl>
                  </SimpleGrid>
                  <FormControl mt={3}><FormLabel fontSize="xs" color="gray.500">Test expectation — edit this before regenerating</FormLabel><Textarea value={brief.expected_output} onChange={update("expected_output")} rows={3} bg="gray.50" fontSize="sm" placeholder="Describe what a correct result must contain" /></FormControl>
                  <HStack mt={3} spacing={3} flexWrap="wrap"><Button size="sm" bg="ink" color="white" _hover={{ bg: "sage.700" }} onClick={runGeneratedTest} isLoading={testingPrompt} loadingText="Testing prompt">Create fixture & run self-test</Button><Button size="sm" variant="outline" onClick={generate} leftIcon={<FiRefreshCw />}>Regenerate with expectation</Button></HStack>
                  {testError && <Text mt={3} color="red.600" fontSize="sm">{testError}</Text>}
                  {promptTest && <Box mt={4} bg={promptTest.passed ? "sage.50" : "coral.50"} border="1px solid" borderColor={promptTest.passed ? "sage.100" : "coral.200"} borderRadius="14px" p={4}><Flex justify="space-between" gap={3} align="start"><Box><Text fontSize="sm" fontWeight="700">{promptTest.passed ? "Self-test passed" : "Self-test needs another pass"}</Text><Text mt={1} fontSize="sm" color="gray.700">{promptTest.rationale}</Text></Box><Stack align="end" spacing={2}><Badge colorScheme={promptTest.passed ? "green" : "red"} borderRadius="full" px={3}>Score {Number(promptTest.score).toFixed(2)}</Badge><Badge colorScheme={promptTest.schema_valid ? "green" : "red"} borderRadius="full" px={3}>Schema {promptTest.schema_valid ? "valid" : "invalid"}</Badge></Stack></Flex><Box as="details" mt={3}><Box as="summary" cursor="pointer" fontSize="xs" fontWeight="700">View generated fixture, schema, and response</Box><Text mt={3} fontSize="xs" fontWeight="700" color="gray.500">FAKE INPUT</Text><Text mt={1} whiteSpace="pre-wrap" fontSize="xs" fontFamily="mono">{promptTest.input}</Text><Text mt={3} fontSize="xs" fontWeight="700" color="gray.500">EXPECTED OUTPUT</Text><Text mt={1} whiteSpace="pre-wrap" fontSize="xs">{promptTest.expected_output}</Text><Text mt={3} fontSize="xs" fontWeight="700" color="gray.500">EXPECTED SCHEMA</Text><Text mt={1} whiteSpace="pre-wrap" fontSize="xs" fontFamily="mono">{JSON.stringify(promptTest.expected_schema, null, 2)}</Text><Text mt={3} fontSize="xs" fontWeight="700" color="gray.500">ACTUAL OUTPUT</Text><Text mt={1} whiteSpace="pre-wrap" fontSize="xs" fontFamily="mono">{promptTest.actual_output}</Text></Box></Box>}
                </Box>
              </Box>}
              {!running && !result && <Box bg="sage.50" borderRadius="14px" p={5} mt={5}><Text fontSize="sm" color="sage.700" fontWeight="600">Your council’s work will appear here</Text><Text fontSize="sm" color="gray.600" mt={1}>Five focused passes, one prompt you can actually use.</Text></Box>}
            </Box>
            <HStack px={2} color="gray.500" fontSize="xs"><Icon as={FiChevronDown} /><Text>Each stage keeps the goal in view, so the final prompt stays grounded in your example.</Text></HStack>
          </Stack>
        </SimpleGrid>
        <Divider my={{ base: 12, md: 20 }} borderColor="blackAlpha.100" />
        <Flex justify="space-between" flexWrap="wrap" gap={4} color="gray.500" fontSize="sm"><Text>Set the expectation. Let the LLMs write the prompt.</Text><Text>Private by default · Nothing is saved</Text></Flex>
      </Container>
    </Box>
  );
}

createRoot(document.getElementById("root")).render(<ChakraProvider theme={theme}><App /></ChakraProvider>);
