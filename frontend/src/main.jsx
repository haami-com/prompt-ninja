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
import { HooksPage } from "./HooksPage";
import "./styles.css";

// In development Vite proxies API requests to the local backend. Deployments can
// still point at a separate API by setting VITE_API_URL.
const API_URL = (import.meta.env.VITE_API_URL || "").replace(/\/$/, "");
const apiFetch = async (path, options) => {
  try {
    return await fetch(`${API_URL}${path}`, options);
  } catch (fetchError) {
    throw new Error("Prompt Ninja’s API is offline. Start the backend on port 8000, then try again.", { cause: fetchError });
  }
};
const apiError = async (response, fallback) => {
  const payload = await response.json().catch(() => null);
  if (payload?.detail) return new Error(payload.detail);
  if (response.status >= 500) return new Error("Prompt Ninja’s API is offline. Start the backend on port 8000, then try again.");
  return new Error(fallback);
};
const describeRunError = (message) => {
  if (message?.includes("invalid_json_schema") || message?.includes("Invalid schema for response_format")) {
    return "The provider rejected the compiler’s output format. You can retry now or change the judge model first.";
  }
  return "The board stopped before it could finish. Retry with the same setup or adjust a model first.";
};
const DEFAULT_MODEL = "google/gemini-2.5-flash";
const DEFAULT_CREATOR_MODELS = ["openai/gpt-5.6-luna", "deepseek/deepseek-v4-flash", "google/gemini-3.5-flash"];
const DEFAULT_JUDGE_MODEL = "openai/gpt-5.6-terra";
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
  const [page, setPage] = useState("board");
  const [requestText, setRequestText] = useState("");
  const [brief, setBrief] = useState(starter);
  const [enhancement, setEnhancement] = useState(null);
  const [files, setFiles] = useState([]);
  const [events, setEvents] = useState([]);
  const [result, setResult] = useState(null);
  const [enhancing, setEnhancing] = useState(false);
  const [running, setRunning] = useState(false);
  const [runError, setRunError] = useState("");
  const [judgeCandidate, setJudgeCandidate] = useState("");
  const [error, setError] = useState("");
  const [copied, setCopied] = useState(false);
  const [downloading, setDownloading] = useState(false);
  const [testModel, setTestModel] = useState(DEFAULT_MODEL);
  const [promptTest, setPromptTest] = useState(null);
  const [testingPrompt, setTestingPrompt] = useState(false);
  const [updatingPrompt, setUpdatingPrompt] = useState(false);
  const [updateFeedback, setUpdateFeedback] = useState("");
  const [testError, setTestError] = useState("");
  const [modelOptions, setModelOptions] = useState([{ id: DEFAULT_MODEL, name: DEFAULT_MODEL, pricing: {} }]);
  const [models, setModels] = useState({ creators: DEFAULT_CREATOR_MODELS, judge: DEFAULT_JUDGE_MODEL });
  const [editingModel, setEditingModel] = useState(null);
  const [editingPrompt, setEditingPrompt] = useState(null);
  const [promptConfig, setPromptConfig] = useState({
    creators: ["", "", ""],
    judge: "",
    metadata: { creators: [null, null, null], judge: null },
  });

  useEffect(() => {
    apiFetch("/api/prompts")
      .then((response) => response.ok ? response.json() : null)
      .then((defaults) => defaults && setPromptConfig(defaults))
      .catch(() => {});
  }, []);

  useEffect(() => {
    apiFetch("/api/models")
      .then((response) => response.ok ? response.json() : null)
      .then((configuration) => {
        if (!configuration?.models?.length || !configuration.default_model) return;
        setModelOptions(configuration.models);
        setModels({
          creators: configuration.creator_models?.length === 3 ? configuration.creator_models : DEFAULT_CREATOR_MODELS,
          judge: configuration.judge_model || DEFAULT_JUDGE_MODEL,
        });
        setTestModel(configuration.creator_models?.[0] || configuration.default_model);
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
      const response = await apiFetch("/api/enhance-brief", { method: "POST", body });
      if (!response.ok) throw await apiError(response, "Prompt Ninja could not enhance the brief.");
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
    setRunning(true); setError(""); setRunError(""); setJudgeCandidate(""); setResult(null); setEvents([]); setPromptTest(null); setTestError("");
    const body = new FormData();
    Object.entries({ ...brief, source_text: enhancement.enhanced_request || "" }).forEach(([key, value]) => body.append(key, value));
    body.append("creator_models", JSON.stringify(models.creators));
    body.append("judge_model", models.judge);
    body.append("creator_prompts", JSON.stringify(promptConfig.creators));
    body.append("judge_prompt", promptConfig.judge);
    files.forEach((file) => body.append("files", file));
    try {
      const response = await apiFetch("/api/generate", { method: "POST", body });
      if (!response.ok) throw await apiError(response, "The Board of Prompts could not start.");
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      const consume = (line) => {
        if (!line.trim()) return;
        const item = JSON.parse(line);
        if (item.type === "agent") {
          setEvents((current) => [...current, item.data]);
          if (item.data?.stage === "synthesis" && item.data?.status === "complete" && item.data?.payload?.final_prompt) {
            setJudgeCandidate(item.data.payload.final_prompt);
          }
        }
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
      setRunError(runError.message || "The Board of Prompts stopped before it could finish.");
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
          prompt: { ...definition.prompt, system: finalPrompt },
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

  async function copyJudgeCandidate() {
    if (!judgeCandidate) return;
    await navigator.clipboard.writeText(judgeCandidate);
    setCopied(true); setTimeout(() => setCopied(false), 1600);
  }

  async function downloadPrompt() {
    if (!result?.final_prompt) return;
    setDownloading(true); setError("");
    try {
      const response = await apiFetch("/api/export-prompt", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ final_prompt: result.final_prompt, goal: brief.outcome, model: models.judge, definition: result.prompt_definition || null }),
      });
      if (!response.ok) throw await apiError(response, "The prompt could not be exported.");
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

  function updateTestExpectation(index, expectedOutput) {
    setResult((current) => {
      if (!current?.prompt_definition) return current;
      const tests = [...(current.prompt_definition.tests || [])];
      tests[index] = { ...tests[index], expected_output: expectedOutput };
      return { ...current, prompt_definition: { ...current.prompt_definition, tests } };
    });
    setPromptTest(null);
  }

  async function runGeneratedTest() {
    if (!result?.prompt_definition) return;
    setTestingPrompt(true); setTestError(""); setPromptTest(null);
    try {
      const response = await apiFetch("/api/test-artifact", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ definition: result.prompt_definition, judge_model: models.judge }),
      });
      if (!response.ok) throw await apiError(response, "The prompt test could not run.");
      setPromptTest(await response.json());
    } catch (testRunError) {
      setTestError(testRunError.message || "The prompt test could not run.");
    } finally { setTestingPrompt(false); }
  }

  async function updateArtifactPrompt() {
    if (!result?.prompt_definition || updateFeedback.trim().length < 3) return;
    setUpdatingPrompt(true); setTestError("");
    try {
      const response = await apiFetch("/api/update-artifact", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ definition: result.prompt_definition, feedback: updateFeedback, model: testModel, judge_model: models.judge }),
      });
      if (!response.ok) throw await apiError(response, "The prompt could not be updated.");
      const updated = await response.json();
      setResult((current) => ({ ...current, final_prompt: updated.definition.prompt.system, prompt_definition: updated.definition }));
      setPromptTest(updated.report);
      setUpdateFeedback("");
    } catch (updateError) {
      setTestError(updateError.message || "The prompt could not be updated.");
    } finally { setUpdatingPrompt(false); }
  }

  const boardStages = [
    { stage: "requirements", icon: FiMessageCircle, label: "Requirements analyst", detail: "Maps the objective, constraints, and edge cases." },
    { stage: "creator_1", icon: FiStar, label: "Creator 1", detail: "Drafts a direct, practical approach.", creatorIndex: 0 },
    { stage: "creator_2", icon: FiStar, label: "Creator 2", detail: "Explores a distinct alternative.", creatorIndex: 1 },
    { stage: "creator_3", icon: FiStar, label: "Creator 3", detail: "Stress-tests the instructions.", creatorIndex: 2 },
    { stage: "synthesis", icon: FiArrowUpRight, label: "Judge", detail: "Selects and combines the strongest ideas.", isJudge: true },
    { stage: "validation", icon: FiCheck, label: "Self-test & compiler", detail: "Validates and packages the final prompt." },
  ];
  const phaseAt = {
    0: ["01", "Define", "Clarify the assignment"],
    1: ["02", "Explore", "Generate competing approaches"],
    4: ["03", "Decide", "Synthesize the best solution"],
    5: ["04", "Validate", "Test and package the result"],
  };

  return (
    <Box className="app-shell">
      <Container maxW="1340px" py={{ base: 5, md: 7 }}>
        <Flex className="nav-shell" justify="space-between" align="center" mb={{ base: 10, md: 16 }}>
          <HStack spacing={3}><IconBox className="brand-mark" boxSize="36px" borderRadius="9px" bg="ink" color="white"><Icon as={GiNinjaHead} boxSize={5} /></IconBox><Text display={{ base: "none", sm: "block" }} fontWeight="700" fontSize="lg" letterSpacing="-0.04em">prompt<Box as="span" color="sage.500">ninja</Box></Text></HStack>
          <HStack spacing={{ base: 1, md: 3 }} color="gray.600" fontSize="sm"><Button size="sm" variant={page === "board" ? "solid" : "ghost"} bg={page === "board" ? "white" : "transparent"} onClick={() => setPage("board")}>Board</Button><Button size="sm" variant={page === "hooks" ? "solid" : "ghost"} bg={page === "hooks" ? "white" : "transparent"} onClick={() => setPage("hooks")}>Hooks</Button><HStack display={{ base: "none", md: "flex" }} spacing={2} border="1px solid" borderColor="blackAlpha.200" bg="whiteAlpha.700" px={3} py={1.5} borderRadius="full"><Box w="7px" h="7px" bg="sage.500" borderRadius="full" boxShadow="0 0 0 3px rgba(15,159,146,.13)" /><Text className="eyebrow" fontSize="10px">Private</Text></HStack></HStack>
        </Flex>

        {page === "hooks" ? <HooksPage apiFetch={apiFetch} /> : <><SimpleGrid columns={{ base: 1, lg: 2 }} gap={{ base: 10, lg: 20 }} alignItems="start">
          <Stack spacing={{ base: 6, md: 8 }}>
            <Box>
              <Text className="eyebrow" fontSize="xs" color="sage.700" fontWeight="600">PROMPT NINJA FORMAT DEMO</Text>
              <Text mt={{ base: 3, md: 4 }} color="gray.600" maxW="570px" fontSize="sm" lineHeight="1.6">This demonstrates a Prompt Ninja `*.prompt.toml` file and the workflow around it: generation, validation, semantic tests, safe updates, and runtime hooks.</Text>
              <Heading className="hero-copy" mt={{ base: 4, md: 6 }} fontSize={{ base: "4xl", md: "6xl" }} lineHeight=".98" letterSpacing="-0.065em">Let LLMs be the <Box as="span" className="hero-accent">prompt engineer.</Box></Heading>
              <Text mt={{ base: 4, md: 6 }} color="gray.700" maxW="570px" fontSize={{ base: "md", md: "lg" }} lineHeight="1.65">Instead of writing the prompt yourself, define the behavior you expect. The Board of Prompts will create and test the prompt for you.</Text>
              <Text mt={3} color="ink" fontWeight="700" fontSize={{ base: "md", md: "lg" }}>Want to try it? Describe something you want to achieve with an LLM.</Text>
            </Box>
            <Stack spacing={{ base: 4, md: 5 }}>
              <FormControl isRequired>
                <FormLabel className="eyebrow" fontSize="xs" fontWeight="700">WHAT SHOULD THE LLM ACHIEVE?</FormLabel>
                <Textarea
                  value={requestText}
                  onChange={changeRequest}
                  placeholder={'Example: Turn release notes into a short customer update. Preserve dates, avoid internal engineering terms, and do not invent details.'}
                  rows={4}
                  bg="white"
                  borderColor="blackAlpha.200"
                />
                <Text mt={2} fontSize="xs" color="gray.500">Describe the outcome and any rules that matter. No prompt-writing or special format needed.</Text>
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
              {!enhancement && <Button onClick={enhanceBrief} isLoading={enhancing} loadingText="Understanding your expectation" size="lg" bg="ink" color="white" _hover={{ bg: "sage.700", boxShadow: "5px 5px 0 #f15b2a" }} rightIcon={<FiArrowUpRight />} borderRadius="10px" py={7}>Create my prompt</Button>}

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
            <Flex justify="space-between" align="end" gap={4}><Box><Text className="eyebrow" fontSize="xs" color="sage.700" fontWeight="700">THE BOARD OF PROMPTS</Text><Heading fontSize={{ base: "xl", md: "2xl" }} lineHeight="1.15" letterSpacing="-0.045em" mt={2}>Specialist agents, one shared objective.</Heading></Box><Stack spacing={2} align="end" flexShrink={0}><Text className="eyebrow" fontSize="10px" color="sage.700">PN // 01</Text><Badge colorScheme={running ? "orange" : result ? "green" : runError ? "red" : "teal"} borderRadius="full" px={3} py={1}>{running ? "In session" : result ? "Complete" : runError ? "Failed" : "Standby"}</Badge></Stack></Flex>
            <Box bg="white" border="1px solid" borderColor="blackAlpha.100" borderRadius="20px" p={{ base: 5, md: 7 }} boxShadow="0 20px 60px rgba(32, 49, 39, .07)">
              <Stack spacing={0}>
                {boardStages.map((agent, index, stages) => {
                  const event = events.find((item) => item.stage === agent.stage && item.status === "complete"); const active = running && !event && (index === 0 || completed.has(stages[index - 1].stage)); const failed = Boolean(runError) && agent.stage === "validation" && !event;
                  const canChangeModel = agent.creatorIndex !== undefined || agent.isJudge;
                  const selectedModel = agent.isJudge ? models.judge : models.creators[agent.creatorIndex];
                  const promptMetadata = agent.isJudge ? promptConfig.metadata?.judge : promptConfig.metadata?.creators?.[agent.creatorIndex];
                  const promptValue = agent.isJudge ? promptConfig.judge : promptConfig.creators[agent.creatorIndex];
                    const phase = phaseAt[index];
                    const resolvedModel = agent.isJudge ? result?.judge_model || selectedModel : agent.creatorIndex !== undefined ? result?.creators?.[agent.creatorIndex]?.model || selectedModel : null;
                    return <React.Fragment key={agent.stage}>{phase && <Flex className="phase-label" align="center" gap={3}><Text>{phase[0]}</Text><Box><Text fontWeight="700">{phase[1]}</Text><Text>{phase[2]}</Text></Box></Flex>}<Box className={`agent-row ${event ? "is-done" : active ? "is-active" : failed ? "is-failed" : ""}`}><Flex gap={4} align="start"><IconBox className="agent-icon" boxSize="36px" borderRadius="10px" bg={event ? "sage.100" : active || failed ? "coral.50" : "gray.100"} color={event ? "sage.700" : active || failed ? "coral.500" : "gray.400"}><Icon as={active ? FiLoader : event ? FiCheck : failed ? FiRefreshCw : agent.icon} /></IconBox><Box flex="1" minW={0}><Flex className="agent-heading" justify="space-between" align="start" gap={3}><Box minW={0}><HStack spacing={2}><Text fontWeight="700" fontSize="sm">{agent.label}</Text>{event && <Badge colorScheme="green" borderRadius="full">done</Badge>}{failed && <Badge colorScheme="red" borderRadius="full">stopped</Badge>}</HStack>{resolvedModel && <Text className="agent-model" title={modelLabel(resolvedModel)}>{modelLabel(resolvedModel)}</Text>}</Box>{canChangeModel && !running && <HStack className="agent-actions" spacing={1}><Button size="xs" variant="ghost" color="sage.700" px={2} h="28px" leftIcon={<FiEdit2 />} onClick={() => { setEditingPrompt(null); setEditingModel((current) => current === agent.stage ? null : agent.stage); }}>{result ? "Next model" : "Model"}</Button><Button size="xs" variant="ghost" color="sage.700" px={2} h="28px" onClick={() => { setEditingModel(null); setEditingPrompt((current) => current === agent.stage ? null : agent.stage); }}>Prompt</Button></HStack>}</Flex><Text fontSize="sm" color={failed ? "red.600" : "gray.500"} mt={2}>{failed ? "Stopped before compilation completed." : event?.summary || agent.detail}</Text>{canChangeModel && editingModel === agent.stage && <Select mt={3} size="sm" value={selectedModel} bg="gray.50" borderColor="blackAlpha.100" aria-label={`Model for ${agent.label}`} onChange={(changeEvent) => { const value = changeEvent.target.value; setModels((current) => agent.isJudge ? { ...current, judge: value } : { ...current, creators: current.creators.map((item, itemIndex) => itemIndex === agent.creatorIndex ? value : item) }); setEditingModel(null); }}>{modelOptions.map((value) => <option key={modelId(value)} value={modelId(value)}>{modelOptionLabel(value)}</option>)}</Select>}{canChangeModel && editingPrompt === agent.stage && <Box mt={3} bg="gray.50" border="1px solid" borderColor="blackAlpha.100" borderRadius="12px" p={3}><Textarea value={promptValue} isDisabled={running} onChange={(promptEvent) => setPromptConfig((current) => agent.isJudge ? { ...current, judge: promptEvent.target.value } : { ...current, creators: current.creators.map((item, itemIndex) => itemIndex === agent.creatorIndex ? promptEvent.target.value : item) })} rows={agent.isJudge ? 6 : 4} bg="white" borderColor="blackAlpha.100" fontSize="sm" aria-label={`Prompt instruction for ${agent.label}`} /><Flex mt={3} justify="space-between" align="start" gap={3} flexWrap="wrap"><Box><Text fontSize="xs" color="gray.500" mb={2}>Runtime variables</Text><HStack spacing={2} flexWrap="wrap">{promptMetadata?.variables?.map((variable) => <Badge key={variable.name} colorScheme={variable.present_in_template ? "green" : "red"} fontFamily="mono">{`{{${variable.name}}}`} · {variable.type}</Badge>) || <Text fontSize="xs" color="gray.400">Loading variable definitions…</Text>}</HStack></Box><Badge colorScheme={promptMetadata?.valid ? "green" : "red"} borderRadius="full" px={3}>{promptMetadata?.valid ? "Variables validated" : `Missing: ${promptMetadata?.missing_variables?.join(", ") || "definitions"}`}</Badge></Flex><Text mt={3} fontSize="xs" color="gray.500">These variables are validated in the PromptNinja TOML template and injected automatically with this instruction.</Text></Box>}</Box></Flex></Box></React.Fragment>;
                })}
              </Stack>
              {runError && !running && <Box bg="coral.50" border="1px solid" borderColor="coral.200" borderRadius="12px" p={4} mt={5}><Flex justify="space-between" align="start" gap={4} flexWrap="wrap"><Box flex="1" minW="220px"><Text fontSize="sm" fontWeight="700">Board stopped during validation</Text><Text mt={1} fontSize="sm" color="gray.700">{describeRunError(runError)}</Text></Box><Button size="sm" bg="ink" color="white" _hover={{ bg: "sage.700" }} leftIcon={<FiRefreshCw />} onClick={generate}>Retry board</Button></Flex><Box as="details" mt={3}><Box as="summary" cursor="pointer" fontSize="xs" color="gray.600" fontWeight="600">Technical details</Box><Text mt={2} fontSize="xs" color="gray.600" fontFamily="mono" whiteSpace="pre-wrap" maxH="160px" overflowY="auto">{runError}</Text></Box></Box>}
              {runError && judgeCandidate && !running && <Box mt={5} pt={5} borderTop="1px solid" borderColor="blackAlpha.100"><Flex justify="space-between" align="center" gap={3} mb={3}><Box><HStack spacing={2}><Text className="eyebrow" fontSize="xs" color="sage.700" fontWeight="700">JUDGE PROMPT</Text><Badge colorScheme="orange" borderRadius="full">unvalidated</Badge></HStack><Text mt={1} fontSize="xs" color="gray.500">The judge completed this prompt before validation stopped.</Text></Box><Button size="sm" variant="ghost" leftIcon={copied ? <FiCheck /> : <FiCopy />} onClick={copyJudgeCandidate}>{copied ? "Copied" : "Copy"}</Button></Flex><Textarea value={judgeCandidate} onChange={(event) => setJudgeCandidate(event.target.value)} minH="240px" resize="vertical" bg="sage.50" borderColor="sage.100" fontSize="sm" lineHeight="1.7" /></Box>}
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
                  <Text fontSize="xs" fontWeight="700" letterSpacing="0.1em" color="sage.700">PROMPT CONTRACT WORKSPACE</Text>
                  <Text mt={2} fontSize="sm" color="gray.600">Edit an embedded expectation, run the complete contract, then revise only the prompt implementation. Updates preserve every test and rerun them automatically.</Text>
                  <SimpleGrid columns={{ base: 1, md: 2 }} gap={3} mt={4}>
                    <FormControl><FormLabel fontSize="xs" color="gray.500">Updater model</FormLabel><Select value={testModel} onChange={(event) => setTestModel(event.target.value)} size="sm" bg="gray.50">{modelOptions.map((value) => <option key={modelId(value)} value={modelId(value)}>{modelOptionLabel(value)}</option>)}</Select></FormControl>
                    <FormControl><FormLabel fontSize="xs" color="gray.500">Judge model</FormLabel><Select value={models.judge} onChange={(event) => setModels((current) => ({ ...current, judge: event.target.value }))} size="sm" bg="gray.50">{modelOptions.map((value) => <option key={modelId(value)} value={modelId(value)}>{modelOptionLabel(value)}</option>)}</Select></FormControl>
                  </SimpleGrid>
                  <Stack spacing={3} mt={4}>{(result.prompt_definition?.tests || []).map((test, index) => <Box key={test.name || index} bg="gray.50" border="1px solid" borderColor="blackAlpha.100" borderRadius="10px" p={3}><Flex justify="space-between" gap={3}><Text fontSize="sm" fontWeight="700">{test.name || `Contract ${index + 1}`}</Text><Badge colorScheme="teal">#{index + 1}</Badge></Flex><Text mt={2} fontSize="xs" color="gray.500">INPUT</Text><Text mt={1} fontSize="xs" fontFamily="mono" whiteSpace="pre-wrap">{JSON.stringify(test.variable, null, 2)}</Text><FormControl mt={3}><FormLabel fontSize="xs" color="gray.500">Expected output</FormLabel><Textarea value={test.expected_output} onChange={(event) => updateTestExpectation(index, event.target.value)} rows={3} bg="white" fontSize="sm" /></FormControl></Box>)}</Stack>
                  <HStack mt={3} spacing={3} flexWrap="wrap"><Button size="sm" bg="ink" color="white" _hover={{ bg: "sage.700" }} onClick={runGeneratedTest} isLoading={testingPrompt} loadingText="Running contract">Run all tests</Button><Badge colorScheme="gray">Threshold {Number(result.prompt_definition?.testing?.pass_threshold ?? 0.95).toFixed(2)}</Badge></HStack>
                  <FormControl mt={5}><FormLabel fontSize="xs" color="gray.500">Prompt revision</FormLabel><Textarea value={updateFeedback} onChange={(event) => setUpdateFeedback(event.target.value)} rows={3} bg="sage.50" borderColor="sage.100" fontSize="sm" placeholder="Describe how the prompt should improve without changing its tests" /></FormControl>
                  <Button mt={3} size="sm" variant="outline" leftIcon={<FiRefreshCw />} onClick={updateArtifactPrompt} isLoading={updatingPrompt} loadingText="Updating and rerunning" isDisabled={updateFeedback.trim().length < 3}>Update prompt & rerun</Button>
                  {testError && <Text mt={3} color="red.600" fontSize="sm">{testError}</Text>}
                  {promptTest && <Box mt={4}><Flex justify="space-between" align="center" gap={3}><Text fontSize="sm" fontWeight="700">{promptTest.passed ? "Contract passed" : "Contract failed"}</Text><Badge colorScheme={promptTest.passed ? "green" : "red"} borderRadius="full" px={3}>{promptTest.results?.filter((item) => item.passed).length || 0}/{promptTest.results?.length || 0} passed</Badge></Flex><Stack spacing={3} mt={3}>{(promptTest.results || []).map((test, index) => <Box key={`${test.name}-${index}`} bg={test.passed ? "sage.50" : "coral.50"} border="1px solid" borderColor={test.passed ? "sage.100" : "coral.200"} borderRadius="10px" p={4}><Flex justify="space-between" gap={3}><Text fontSize="sm" fontWeight="700">{test.name}</Text><Badge colorScheme={test.passed ? "green" : "red"}>{test.score == null ? (test.passed ? "pass" : "error") : Number(test.score).toFixed(2)}</Badge></Flex><Text mt={2} fontSize="sm" color="gray.700" whiteSpace="pre-wrap">{test.rationale || test.error}</Text>{test.prompt_suggestion && <Box mt={3} bg="whiteAlpha.700" p={3} borderRadius="8px"><Text fontSize="xs" fontWeight="700" color="sage.700">PROMPT SUGGESTION</Text><Text mt={1} fontSize="sm">{test.prompt_suggestion}</Text><Button mt={2} size="xs" variant="outline" onClick={() => setUpdateFeedback(test.prompt_suggestion)}>Use suggestion</Button></Box>}{test.test_suggestion && <Box mt={3} bg="whiteAlpha.700" p={3} borderRadius="8px"><Text fontSize="xs" fontWeight="700" color="coral.500">TEST SUGGESTION</Text><Text mt={1} fontSize="sm">{test.test_suggestion}</Text></Box>}<Box as="details" mt={3}><Box as="summary" cursor="pointer" fontSize="xs" fontWeight="700">Full input, expectation, and response</Box><Text mt={3} fontSize="xs" fontWeight="700" color="gray.500">INPUT</Text><Text mt={1} whiteSpace="pre-wrap" fontSize="xs" fontFamily="mono">{JSON.stringify(test.input, null, 2)}</Text><Text mt={3} fontSize="xs" fontWeight="700" color="gray.500">EXPECTED</Text><Text mt={1} whiteSpace="pre-wrap" fontSize="xs">{typeof test.expected === "string" ? test.expected : JSON.stringify(test.expected, null, 2)}</Text><Text mt={3} fontSize="xs" fontWeight="700" color="gray.500">ACTUAL</Text><Text mt={1} whiteSpace="pre-wrap" fontSize="xs" fontFamily="mono">{typeof test.actual === "string" ? test.actual : JSON.stringify(test.actual, null, 2)}</Text></Box></Box>)}</Stack></Box>}
                </Box>
              </Box>}
              {!running && !result && !runError && <Box bg="sage.50" borderRadius="14px" p={5} mt={5}><Text fontSize="sm" color="sage.700" fontWeight="600">Your board’s work will appear here</Text><Text fontSize="sm" color="gray.600" mt={1}>Six focused passes, one validated prompt you can actually use.</Text></Box>}
            </Box>
            <HStack px={2} color="gray.500" fontSize="xs"><Icon as={FiChevronDown} /><Text>Each stage keeps the goal in view, so the final prompt stays grounded in your example.</Text></HStack>
          </Stack>
        </SimpleGrid>
        <Divider my={{ base: 12, md: 20 }} borderColor="blackAlpha.100" />
        <Flex justify="space-between" flexWrap="wrap" gap={4} color="gray.500" fontSize="sm"><Text>Set the expectation. Let the LLMs write the prompt.</Text><Text>Private by default · Nothing is saved</Text></Flex></>}
      </Container>
    </Box>
  );
}

createRoot(document.getElementById("root")).render(<ChakraProvider theme={theme}><App /></ChakraProvider>);
