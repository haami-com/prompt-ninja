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
import { FiArrowUpRight, FiCheck, FiChevronDown, FiCopy, FiDownload, FiEdit2, FiFileText, FiLoader, FiMessageCircle, FiRefreshCw, FiStar, FiUploadCloud } from "react-icons/fi";
import { GiNinjaHead } from "react-icons/gi";
import { theme } from "./theme";
import "./styles.css";

// 127.0.0.1 avoids resolving `localhost` to a different IPv6/container listener.
const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";
const DEFAULT_MODEL = "google/gemini-2.5-flash";
const modelId = (model) => typeof model === "string" ? model : model?.id;
const modelLabel = (model) => String(typeof model === "string" ? model : model?.name || model?.id || "Unknown model");
const pricePerMillion = (model, kind) => {
  const price = Number(model?.pricing?.[kind]);
  return Number.isFinite(price) ? `$${(price * 1_000_000).toFixed(price * 1_000_000 < 0.01 ? 4 : 2)}/M ${kind === "prompt" ? "in" : "out"}` : null;
};
const modelOptionLabel = (model) => {
  const pricing = [pricePerMillion(model, "prompt"), pricePerMillion(model, "completion")].filter(Boolean).join(" · ");
  return pricing ? `${modelLabel(model)} — ${pricing}` : modelLabel(model);
};

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
  const [requestText, setRequestText] = useState("");
  const [brief, setBrief] = useState(starter);
  const [enhancement, setEnhancement] = useState(null);
  const [files, setFiles] = useState([]);
  const [events, setEvents] = useState([]);
  const [result, setResult] = useState(null);
  const [enhancing, setEnhancing] = useState(false);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState("");
  const [copied, setCopied] = useState(false);
  const [downloading, setDownloading] = useState(false);
  const [testModel, setTestModel] = useState(DEFAULT_MODEL);
  const [promptTest, setPromptTest] = useState(null);
  const [testingPrompt, setTestingPrompt] = useState(false);
  const [testError, setTestError] = useState("");
  const [modelOptions, setModelOptions] = useState([{ id: DEFAULT_MODEL, name: DEFAULT_MODEL, pricing: {} }]);
  const [models, setModels] = useState({ creators: [DEFAULT_MODEL, DEFAULT_MODEL, DEFAULT_MODEL], judge: DEFAULT_MODEL });
  const [editingModel, setEditingModel] = useState(null);
  const [editingPrompt, setEditingPrompt] = useState(null);
  const [promptConfig, setPromptConfig] = useState({
    creators: ["", "", ""],
    judge: "",
    metadata: { creators: [null, null, null], judge: null },
  });

  useEffect(() => {
    fetch(`${API_URL}/api/prompts`)
      .then((response) => response.ok ? response.json() : null)
      .then((defaults) => defaults && setPromptConfig(defaults))
      .catch(() => {});
  }, []);

  useEffect(() => {
    fetch(`${API_URL}/api/models`)
      .then((response) => response.ok ? response.json() : null)
      .then((configuration) => {
        if (!configuration?.models?.length || !configuration.default_model) return;
        setModelOptions(configuration.models);
        setModels({ creators: Array(3).fill(configuration.default_model), judge: configuration.default_model });
        setTestModel(configuration.default_model);
      })
      .catch(() => {});
  }, []);

  const update = (key) => (event) => setBrief((current) => ({ ...current, [key]: event.target.value }));
  const completed = useMemo(() => new Set(events.filter((event) => event.status === "complete").map((event) => event.stage)), [events]);

  function changeRequest(event) {
    setRequestText(event.target.value);
    setEnhancement(null);
  }

  function changeFiles(event) {
    setFiles(Array.from(event.target.files || []).slice(0, 5));
    setEnhancement(null);
  }

  async function enhanceBrief() {
    if (requestText.trim().length < 8) {
      setError("Describe what you need in at least 8 characters.");
      return;
    }
    setEnhancing(true); setError(""); setResult(null); setEvents([]); setPromptTest(null); setTestError("");
    const body = new FormData();
    body.append("request_text", requestText);
    files.forEach((file) => body.append("files", file));
    try {
      const response = await fetch(`${API_URL}/api/enhance-brief`, { method: "POST", body });
      if (!response.ok) throw new Error((await response.json()).detail || "Prompt Ninja could not enhance the brief.");
      const enhanced = await response.json();
      setEnhancement(enhanced);
      setBrief({
        outcome: enhanced.outcome || "",
        context: enhanced.context || "",
        source_text: "",
        expected_output: enhanced.expected_output || "",
        constraints: enhanced.constraints || "",
      });
    } catch (enhanceError) {
      setError(enhanceError.message || "Prompt Ninja could not enhance the brief.");
    } finally { setEnhancing(false); }
  }

  async function generate() {
    if (!enhancement) {
      setError("Enhance and review the brief before sending it to the board.");
      return;
    }
    setRunning(true); setError(""); setResult(null); setEvents([]); setPromptTest(null); setTestError("");
    const body = new FormData();
    Object.entries({ ...brief, source_text: enhancement.enhanced_request || "" }).forEach(([key, value]) => body.append(key, value));
    body.append("creator_models", JSON.stringify(models.creators));
    body.append("judge_model", models.judge);
    body.append("creator_prompts", JSON.stringify(promptConfig.creators));
    body.append("judge_prompt", promptConfig.judge);
    files.forEach((file) => body.append("files", file));
    try {
      const response = await fetch(`${API_URL}/api/generate`, { method: "POST", body });
      if (!response.ok) throw new Error((await response.json()).detail || "The Board of Prompts could not start.");
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      const consume = (line) => {
        if (!line.trim()) return;
        const item = JSON.parse(line);
        if (item.type === "agent") setEvents((current) => [...current, item.data]);
        if (item.type === "result") setResult(item.data);
        if (item.type === "error") throw new Error(item.message || "The Board of Prompts failed while running.");
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

  function updateFinalPrompt(event) {
    const finalPrompt = event.target.value;
    setResult((current) => {
      if (!current) return current;
      const definition = current.prompt_definition || {};
      return {
        ...current,
        final_prompt: finalPrompt,
        prompt_definition: {
          ...definition,
          template: { ...definition.template, system: finalPrompt },
        },
      };
    });
    setPromptTest(null);
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
        body: JSON.stringify({ final_prompt: result.final_prompt, goal: brief.outcome, model: models.judge, definition: result.prompt_definition || null }),
      });
      if (!response.ok) throw new Error((await response.json()).detail || "The prompt could not be exported.");
      const disposition = response.headers.get("Content-Disposition") || "";
      const filename = disposition.match(/filename="?([^";]+)"?/)?.[1] || "generated-prompt.prompt.toml";
      const url = URL.createObjectURL(await response.blob());
      const link = document.createElement("a");
      link.href = url; link.download = filename; document.body.appendChild(link); link.click(); link.remove();
      window.setTimeout(() => URL.revokeObjectURL(url), 0);
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
        body: JSON.stringify({ final_prompt: result.final_prompt, goal: brief.outcome, context: brief.context, expected_output: brief.expected_output, model: testModel, judge_model: models.judge, definition: result.prompt_definition || null }),
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
          <HStack spacing={3}><IconBox boxSize="34px" borderRadius="12px" bg="ink" color="white"><Icon as={GiNinjaHead} boxSize={5} /></IconBox><Text fontWeight="700" letterSpacing="-0.02em">prompt <Box as="span" color="sage.500">ninja</Box></Text></HStack>
          <HStack spacing={3} color="gray.500" fontSize="sm"><Text>Private workspace</Text><Box w="6px" h="6px" bg="sage.500" borderRadius="full" /></HStack>
        </Flex>

        <SimpleGrid columns={{ base: 1, lg: 2 }} gap={{ base: 10, lg: 20 }} alignItems="start">
          <Stack spacing={8}>
            <Box>
              <Heading mt={4} fontSize={{ base: "3xl", md: "5xl" }} lineHeight="1.04" letterSpacing="-0.055em">You describe<br />the <Box as="span" color="sage.500">expectation.</Box></Heading>
              <Text mt={5} color="gray.600" maxW="500px" fontSize="lg" lineHeight="1.6">The LLMs create the prompt. Share what you want to happen, and let them work out the instructions.</Text>
            </Box>
            <Stack spacing={5}>
              <FormControl isRequired>
                <FormLabel fontSize="sm" fontWeight="700">Tell Prompt Ninja what you need</FormLabel>
                <Textarea
                  value={requestText}
                  onChange={changeRequest}
                  placeholder={'Write naturally. Include the goal, where it will be used, the result you want, and any guardrails.\n\nExample: Use File #1 as the source and File #2 as the tone reference. Create a concise weekly update with decisions, owners, and due dates. Do not invent missing details.'}
                  rows={9}
                  bg="white"
                  borderColor="blackAlpha.200"
                />
                <Text mt={2} fontSize="xs" color="gray.500">No special format needed. Refer to uploads by their numbered labels.</Text>
              </FormControl>
              <Flex className="upload-zone" as="label" htmlFor="files" cursor="pointer" align="center" gap={3}>
                <Icon as={FiUploadCloud} color="sage.500" boxSize={5} />
                <Box>
                  <Text fontSize="sm" fontWeight="700">{files.length ? "Replace reference files" : "Add reference files"}</Text>
                  <Text fontSize="xs" color="gray.500">PDF, DOCX, TXT, MD or CSV · up to 5 files</Text>
                </Box>
                <Input id="files" type="file" accept=".pdf,.docx,.txt,.md,.csv" multiple display="none" onChange={changeFiles} />
              </Flex>
              {files.length > 0 && <Stack spacing={2}>
                {files.map((file, index) => <Flex key={`${file.name}-${file.lastModified}-${index}`} align="center" gap={3} bg="white" border="1px solid" borderColor="blackAlpha.100" borderRadius="12px" px={3} py={2}>
                  <Badge colorScheme="green" borderRadius="full" px={2}>File #{index + 1}</Badge>
                  <Icon as={FiFileText} color="gray.500" />
                  <Text fontSize="sm" color="gray.700" noOfLines={1}>{file.name}</Text>
                </Flex>)}
              </Stack>}
              {!enhancement && <Button onClick={enhanceBrief} isLoading={enhancing} loadingText="Structuring your brief" size="lg" bg="ink" color="white" _hover={{ bg: "sage.700" }} rightIcon={<FiArrowUpRight />} borderRadius="14px" py={7}>Enhance & review</Button>}

              {enhancement && <>
                <Box bg="white" border="1px solid" borderColor="sage.100" borderRadius="18px" p={{ base: 4, md: 5 }} boxShadow="0 14px 40px rgba(32, 49, 39, .06)">
                  <Flex justify="space-between" align="start" gap={3} mb={5}>
                    <Box><Text fontSize="sm" fontWeight="700">Review the enhanced brief</Text><Text mt={1} fontSize="xs" color="gray.500">Edit anything below. The board only starts after you confirm.</Text></Box>
                    <Badge colorScheme="green" borderRadius="full" px={3}>Ready to review</Badge>
                  </Flex>
                  <Stack spacing={4}>
                    <FormControl><FormLabel fontSize="xs" color="gray.500">Polished request</FormLabel><Textarea value={enhancement.enhanced_request} onChange={(event) => setEnhancement((current) => ({ ...current, enhanced_request: event.target.value }))} rows={5} bg="sage.50" borderColor="sage.100" fontSize="sm" /></FormControl>
                    <FormControl isRequired><FormLabel fontSize="xs" color="gray.500">Outcome</FormLabel><Textarea value={brief.outcome} onChange={update("outcome")} rows={3} bg="gray.50" borderColor="blackAlpha.100" fontSize="sm" /></FormControl>
                    <FormControl><FormLabel fontSize="xs" color="gray.500">Usage context</FormLabel><Input value={brief.context} onChange={update("context")} bg="gray.50" borderColor="blackAlpha.100" fontSize="sm" /></FormControl>
                    <FormControl><FormLabel fontSize="xs" color="gray.500">Expected output</FormLabel><Textarea value={brief.expected_output} onChange={update("expected_output")} rows={3} bg="gray.50" borderColor="blackAlpha.100" fontSize="sm" /></FormControl>
                    <FormControl><FormLabel fontSize="xs" color="gray.500">Guardrails</FormLabel><Textarea value={brief.constraints} onChange={update("constraints")} rows={2} bg="gray.50" borderColor="blackAlpha.100" fontSize="sm" /></FormControl>
                    {enhancement.file_references?.length > 0 && <HStack spacing={2} flexWrap="wrap"><Text fontSize="xs" color="gray.500">Referenced:</Text>{enhancement.file_references.map((label) => <Badge key={label} colorScheme="green">{label}</Badge>)}</HStack>}
                  </Stack>
                </Box>
                <HStack spacing={3}>
                  <Button flex="1" variant="outline" onClick={() => setEnhancement(null)} size="lg" borderRadius="14px">Edit original</Button>
                  <Button flex="2" onClick={generate} isLoading={running} loadingText="Board is thinking" size="lg" bg="ink" color="white" _hover={{ bg: "sage.700" }} rightIcon={<FiArrowUpRight />} borderRadius="14px">Send to Board of Prompts</Button>
                </HStack>
              </>}
              {error && <Text color="red.600" fontSize="sm">{error}</Text>}
            </Stack>
          </Stack>

          <Stack className="council-panel" spacing={5}>
            <Flex justify="space-between" align="center"><Box><Text fontSize="xs" color="gray.500" fontWeight="700" letterSpacing="0.1em">THE BOARD OF PROMPTS</Text><Heading fontSize="2xl" letterSpacing="-0.03em" mt={1}>Tell the LLMs what you want. Let them decide what works best.</Heading></Box><Badge colorScheme={running ? "orange" : result ? "green" : "gray"} borderRadius="full" px={3}>{running ? "In session" : result ? "Complete" : "Waiting"}</Badge></Flex>
            <Box bg="white" border="1px solid" borderColor="blackAlpha.100" borderRadius="20px" p={{ base: 5, md: 7 }} boxShadow="0 20px 60px rgba(32, 49, 39, .07)">
              <Stack spacing={0}>
                {[{ stage: "requirements", icon: FiMessageCircle, label: "Requirements analyst", detail: "Mapping the expectation and edge cases" }, { stage: "creator_1", icon: FiStar, label: `Creator 1 · ${modelLabel(result?.creators?.[0]?.model || models.creators[0])}`, detail: "Writing a first proposal", creatorIndex: 0 }, { stage: "creator_2", icon: FiStar, label: `Creator 2 · ${modelLabel(result?.creators?.[1]?.model || models.creators[1])}`, detail: "Trying another approach", creatorIndex: 1 }, { stage: "creator_3", icon: FiStar, label: `Creator 3 · ${modelLabel(result?.creators?.[2]?.model || models.creators[2])}`, detail: "Stress-testing the instructions", creatorIndex: 2 }, { stage: "synthesis", icon: FiArrowUpRight, label: `Judge · ${modelLabel(result?.judge_model || models.judge)}`, detail: "Combining the useful ideas into one prompt", isJudge: true }, { stage: "validation", icon: FiCheck, label: "Self-test & compiler", detail: "Validating and packaging the final prompt" }].map((agent, index, stages) => {
                  const event = events.find((item) => item.stage === agent.stage && item.status === "complete"); const active = running && !event && (index === 0 || completed.has(stages[index - 1].stage));
                  const canChangeModel = agent.creatorIndex !== undefined || agent.isJudge;
                  const selectedModel = agent.isJudge ? models.judge : models.creators[agent.creatorIndex];
                  const promptMetadata = agent.isJudge ? promptConfig.metadata?.judge : promptConfig.metadata?.creators?.[agent.creatorIndex];
                  const promptValue = agent.isJudge ? promptConfig.judge : promptConfig.creators[agent.creatorIndex];
                    return <Box key={agent.stage} className={`agent-row ${event ? "is-done" : active ? "is-active" : ""}`}><Flex gap={4} align="start"><IconBox boxSize="34px" borderRadius="12px" bg={event ? "sage.100" : active ? "coral.50" : "gray.100"} color={event ? "sage.700" : active ? "coral.500" : "gray.400"}><Icon as={active ? FiLoader : event ? FiCheck : agent.icon} /></IconBox><Box flex="1"><Flex justify="space-between" gap={3}><HStack spacing={2} align="center" flexWrap="wrap"><Text fontWeight="700" fontSize="sm">{agent.label}</Text>{canChangeModel && !running && <><Button size="xs" variant="ghost" color="sage.700" px={2} h="24px" leftIcon={<FiEdit2 />} onClick={() => { setEditingPrompt(null); setEditingModel((current) => current === agent.stage ? null : agent.stage); }}>{result ? "Next model" : "Change model"}</Button><Button size="xs" variant="ghost" color="sage.700" px={2} h="24px" onClick={() => { setEditingModel(null); setEditingPrompt((current) => current === agent.stage ? null : agent.stage); }}>Edit prompt</Button></>}</HStack>{event && <Text fontSize="xs" color="sage.700">done</Text>}</Flex><Text fontSize="sm" color="gray.500" mt={1}>{event?.summary || agent.detail}</Text>{canChangeModel && editingModel === agent.stage && <Select mt={3} size="sm" value={selectedModel} bg="gray.50" borderColor="blackAlpha.100" aria-label={`Model for ${agent.label}`} onChange={(changeEvent) => { const value = changeEvent.target.value; setModels((current) => agent.isJudge ? { ...current, judge: value } : { ...current, creators: current.creators.map((item, itemIndex) => itemIndex === agent.creatorIndex ? value : item) }); setEditingModel(null); }}>{modelOptions.map((value) => <option key={modelId(value)} value={modelId(value)}>{modelOptionLabel(value)}</option>)}</Select>}{canChangeModel && editingPrompt === agent.stage && <Box mt={3} bg="gray.50" border="1px solid" borderColor="blackAlpha.100" borderRadius="12px" p={3}><Textarea value={promptValue} isDisabled={running} onChange={(promptEvent) => setPromptConfig((current) => agent.isJudge ? { ...current, judge: promptEvent.target.value } : { ...current, creators: current.creators.map((item, itemIndex) => itemIndex === agent.creatorIndex ? promptEvent.target.value : item) })} rows={agent.isJudge ? 6 : 4} bg="white" borderColor="blackAlpha.100" fontSize="sm" aria-label={`Prompt instruction for ${agent.label}`} /><Flex mt={3} justify="space-between" align="start" gap={3} flexWrap="wrap"><Box><Text fontSize="xs" color="gray.500" mb={2}>Runtime variables</Text><HStack spacing={2} flexWrap="wrap">{promptMetadata?.variables?.map((variable) => <Badge key={variable.name} colorScheme={variable.present_in_template ? "green" : "red"} fontFamily="mono">{`{{${variable.name}}}`} · {variable.type}</Badge>) || <Text fontSize="xs" color="gray.400">Loading variable definitions…</Text>}</HStack></Box><Badge colorScheme={promptMetadata?.valid ? "green" : "red"} borderRadius="full" px={3}>{promptMetadata?.valid ? "Variables validated" : `Missing: ${promptMetadata?.missing_variables?.join(", ") || "definitions"}`}</Badge></Flex><Text mt={3} fontSize="xs" color="gray.500">These variables are validated in the PromptNinja TOML template and injected automatically with this instruction.</Text></Box>}</Box></Flex>{index < stages.length - 1 && <Box className="agent-line" />}</Box>;
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
                <Flex justify="space-between" align="center" mb={3} gap={2}><Text fontSize="xs" fontWeight="700" letterSpacing="0.1em" color="sage.700">FINAL PROMPT</Text><HStack spacing={1}><Button size="sm" variant="ghost" leftIcon={copied ? <FiCheck /> : <FiCopy />} onClick={copyPrompt}>{copied ? "Copied" : "Copy"}</Button><Button size="sm" variant="ghost" leftIcon={<FiDownload />} onClick={downloadPrompt} isLoading={downloading}>Download TOML</Button></HStack></Flex><Textarea value={result.final_prompt} onChange={updateFinalPrompt} minH="260px" resize="vertical" bg="sage.50" borderColor="sage.100" fontSize="sm" lineHeight="1.7" />
                <Box mt={5} pt={5} borderTop="1px solid" borderColor="blackAlpha.100">
                  <Text fontSize="xs" fontWeight="700" letterSpacing="0.1em" color="sage.700">SELF-TEST THE GENERATED PROMPT</Text>
                  <Text mt={2} fontSize="sm" color="gray.600">Prompt Ninja creates a realistic fixture, runs this prompt with the selected model, then asks the judge whether the response meets the expectation.</Text>
                  <SimpleGrid columns={{ base: 1, md: 2 }} gap={3} mt={4}>
                    <FormControl><FormLabel fontSize="xs" color="gray.500">Runner model</FormLabel><Select value={testModel} onChange={(event) => setTestModel(event.target.value)} size="sm" bg="gray.50">{modelOptions.map((value) => <option key={modelId(value)} value={modelId(value)}>{modelOptionLabel(value)}</option>)}</Select></FormControl>
                    <FormControl><FormLabel fontSize="xs" color="gray.500">Judge model</FormLabel><Select value={models.judge} onChange={(event) => setModels((current) => ({ ...current, judge: event.target.value }))} size="sm" bg="gray.50">{modelOptions.map((value) => <option key={modelId(value)} value={modelId(value)}>{modelOptionLabel(value)}</option>)}</Select></FormControl>
                  </SimpleGrid>
                  <FormControl mt={3}><FormLabel fontSize="xs" color="gray.500">Test expectation — edit this before regenerating</FormLabel><Textarea value={brief.expected_output} onChange={update("expected_output")} rows={3} bg="gray.50" fontSize="sm" placeholder="Describe what a correct result must contain" /></FormControl>
                  <HStack mt={3} spacing={3} flexWrap="wrap"><Button size="sm" bg="ink" color="white" _hover={{ bg: "sage.700" }} onClick={runGeneratedTest} isLoading={testingPrompt} loadingText="Testing prompt">Create fixture & run self-test</Button><Button size="sm" variant="outline" onClick={generate} leftIcon={<FiRefreshCw />}>Regenerate with expectation</Button></HStack>
                  {testError && <Text mt={3} color="red.600" fontSize="sm">{testError}</Text>}
                  {promptTest && <Box mt={4} bg={promptTest.passed ? "sage.50" : "coral.50"} border="1px solid" borderColor={promptTest.passed ? "sage.100" : "coral.200"} borderRadius="14px" p={4}><Flex justify="space-between" gap={3} align="start"><Box><Text fontSize="sm" fontWeight="700">{promptTest.passed ? "Self-test passed" : "Self-test needs another pass"}</Text><Text mt={1} fontSize="sm" color="gray.700">{promptTest.rationale}</Text></Box><Badge colorScheme={promptTest.passed ? "green" : "red"} borderRadius="full" px={3}>Score {Number(promptTest.score).toFixed(2)}</Badge></Flex><Box as="details" mt={3}><Box as="summary" cursor="pointer" fontSize="xs" fontWeight="700">View generated fixture and response</Box><Text mt={3} fontSize="xs" fontWeight="700" color="gray.500">FAKE INPUT</Text><Text mt={1} whiteSpace="pre-wrap" fontSize="xs" fontFamily="mono">{promptTest.input}</Text><Text mt={3} fontSize="xs" fontWeight="700" color="gray.500">EXPECTED OUTPUT</Text><Text mt={1} whiteSpace="pre-wrap" fontSize="xs">{promptTest.expected_output}</Text><Text mt={3} fontSize="xs" fontWeight="700" color="gray.500">ACTUAL OUTPUT</Text><Text mt={1} whiteSpace="pre-wrap" fontSize="xs" fontFamily="mono">{promptTest.actual_output}</Text></Box></Box>}
                </Box>
              </Box>}
              {!running && !result && <Box bg="sage.50" borderRadius="14px" p={5} mt={5}><Text fontSize="sm" color="sage.700" fontWeight="600">Your board’s work will appear here</Text><Text fontSize="sm" color="gray.600" mt={1}>Six focused passes, one validated prompt you can actually use.</Text></Box>}
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
