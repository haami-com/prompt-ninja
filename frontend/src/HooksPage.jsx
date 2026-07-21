import React, { useCallback, useEffect, useState } from "react";
import {
  Badge,
  Box,
  Button,
  Flex,
  Heading,
  HStack,
  Icon,
  SimpleGrid,
  Spinner,
  Stack,
  Text,
} from "@chakra-ui/react";
import { FiActivity, FiArrowRight, FiBarChart2, FiEye, FiRefreshCw, FiZap } from "react-icons/fi";

const scoreColor = (score) => score >= 0.8 ? "green" : score >= 0.5 ? "orange" : "red";
const formatTime = (value) => value ? new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "short" }).format(new Date(value)) : "Just now";
const formatTokens = (value) => new Intl.NumberFormat().format(value || 0);
const formatCost = (value) => value == null ? "Pricing unavailable" : `$${value.toFixed(value < 0.01 ? 6 : 4)}`;

export function HooksPage({ apiFetch }) {
  const [feed, setFeed] = useState(null);
  const [error, setError] = useState("");
  const [refreshing, setRefreshing] = useState(false);

  const load = useCallback(async (quiet = false) => {
    if (!quiet) setRefreshing(true);
    try {
      const response = await apiFetch("/api/hooks");
      if (!response.ok) throw new Error("Could not load hook activity.");
      setFeed(await response.json());
      setError("");
    } catch (loadError) {
      setError(loadError.message || "Could not load hook activity.");
    } finally {
      if (!quiet) setRefreshing(false);
    }
  }, [apiFetch]);

  useEffect(() => {
    load();
    const timer = window.setInterval(() => load(true), 5000);
    return () => window.clearInterval(timer);
  }, [load]);

  const quality = feed?.quality;
  const usage = feed?.usage;
  const evaluations = quality?.evaluations || [];
  const usageRecords = usage?.records || [];
  const usageSummary = usage?.summary || {};

  return <Box pb={{ base: 10, md: 16 }}>
    <Flex justify="space-between" align={{ base: "start", md: "end" }} gap={5} flexDirection={{ base: "column", md: "row" }}>
      <Box>
        <Text className="eyebrow" fontSize="xs" color="sage.700" fontWeight="700">// RUNTIME HOOKS</Text>
        <Heading mt={3} fontSize={{ base: "3xl", md: "5xl" }} letterSpacing="-0.06em" lineHeight="1">What happens after a prompt runs?</Heading>
        <Text mt={4} color="gray.600" maxW="720px" lineHeight="1.7">Hooks observe the prompt lifecycle and react to it without changing your prompt call. Use them for evaluation, telemetry, logging, or any side effect that should stay separate from prompt behavior.</Text>
      </Box>
      <Button variant="outline" leftIcon={<FiRefreshCw />} onClick={() => load()} isLoading={refreshing}>Refresh</Button>
    </Flex>

    <SimpleGrid columns={{ base: 1, md: 3 }} gap={4} mt={10}>
      <LifecycleStep number="01" icon={FiZap} title="Your prompt runs" detail="Prompt Ninja renders the artifact, calls the configured model, and returns the response normally." />
      <LifecycleStep number="02" icon={FiEye} title="Hooks observe the event" detail="Each hook receives structured request, response, error, and usage data from the same lifecycle." />
      <LifecycleStep number="03" icon={FiArrowRight} title="Work continues separately" detail="Hooks evaluate or record the run without rewriting the prompt or changing the caller's result." />
    </SimpleGrid>

    {error && <Box mt={6} bg="coral.50" border="1px solid" borderColor="coral.200" borderRadius="12px" p={4}><Text color="red.600" fontSize="sm">{error}</Text></Box>}
    {!feed && !error && <Flex py={20} justify="center"><Spinner color="sage.500" /></Flex>}

    {feed && <>
      <HookHeader number="01" title="Semantic quality" description="Creator 1 · asynchronous LLM judge" badge="Non-blocking" icon={FiActivity} />
      <HookContract
        receives="Rendered system prompt, input, and model output"
        does="Asks a judge model to score the response and explain its decision"
        impact="The caller gets its response immediately; evaluation finishes in the background"
      />
      <Box bg="white" border="1px solid" borderColor="blackAlpha.200" borderRadius="12px" p={{ base: 5, md: 6 }}>
        <Flex justify="space-between" gap={4} align="center"><Box><Text fontWeight="700">Live quality activity</Text><Text mt={1} fontSize="sm" color="gray.500">{quality.completed_calls} Creator 1 calls observed · {evaluations.length} evaluations stored</Text></Box><Badge colorScheme="teal" borderRadius="full" px={3} py={1}>{quality.pending ? `${quality.pending} pending` : "Caught up"}</Badge></Flex>
      </Box>
      {evaluations.length === 0 ? <EmptyState icon={FiActivity} title="Waiting for a quality evaluation" detail="Run the Board once. Creator 1's judged result will appear here automatically." /> : <Stack mt={5} spacing={4}>{evaluations.map((evaluation) => <Box key={evaluation.run_id} bg="white" border="1px solid" borderColor="blackAlpha.200" borderRadius="12px" p={{ base: 5, md: 6 }} boxShadow="5px 5px 0 rgba(15,159,146,.08)"><Flex justify="space-between" align="start" gap={4}><Box minW={0}><HStack spacing={2} flexWrap="wrap"><Text fontWeight="700">{evaluation.prompt_name}</Text><Badge colorScheme={scoreColor(evaluation.score)} borderRadius="full">Score {evaluation.score.toFixed(2)}</Badge></HStack><Text mt={1} className="agent-model">{evaluation.model}</Text></Box><Text fontSize="xs" color="gray.500" flexShrink={0}>{formatTime(evaluation.evaluated_at)}</Text></Flex><Text mt={5} fontSize="sm" color="gray.700" lineHeight="1.7">{evaluation.rationale}</Text><Box as="details" mt={5}><Box as="summary" cursor="pointer" fontSize="sm" fontWeight="700" color="sage.700">Inspect prompt, input, and output</Box><Stack mt={4} spacing={4}><Evidence label="SYSTEM PROMPT" value={evaluation.system_prompt} /><Evidence label="INPUT" value={evaluation.input} /><Evidence label="OUTPUT" value={evaluation.output} /></Stack></Box></Box>)}</Stack>}

      <HookHeader number="02" title="Usage & cost" description="Creator 2 · passive provider telemetry" badge="No extra LLM call" icon={FiBarChart2} />
      <HookContract
        receives="Provider token counts and the model used for the response"
        does="Records input, output, and total tokens, then estimates cost from catalog pricing"
        impact="No model is called and the response is not inspected or modified"
      />
      <SimpleGrid columns={{ base: 1, md: 3 }} gap={4}>
        <UsageStat label="Input tokens" value={formatTokens(usageSummary.input_tokens)} />
        <UsageStat label="Output tokens" value={formatTokens(usageSummary.output_tokens)} />
        <UsageStat label="Estimated cost" value={formatCost(usageSummary.total_cost)} />
      </SimpleGrid>
      {usageRecords.length === 0 ? <EmptyState icon={FiBarChart2} title="Waiting for usage telemetry" detail="Run the Board once. Creator 2 token counts and estimated cost will appear here." /> : <Stack mt={5} spacing={3}>{usageRecords.map((record) => <Flex key={record.run_id} bg="white" border="1px solid" borderColor="blackAlpha.200" borderRadius="12px" p={{ base: 4, md: 5 }} gap={4} align={{ base: "start", md: "center" }} justify="space-between" flexDirection={{ base: "column", md: "row" }}><Box minW={0}><Text fontWeight="700" fontSize="sm">{record.prompt_name}</Text><Text className="agent-model">{record.model}</Text></Box><HStack spacing={{ base: 4, md: 8 }} align="start"><UsageDatum label="Input" value={formatTokens(record.input_tokens)} /><UsageDatum label="Output" value={formatTokens(record.output_tokens)} /><UsageDatum label="Total" value={formatTokens(record.total_tokens)} /><UsageDatum label="Cost" value={formatCost(record.total_cost)} /></HStack><Text fontSize="xs" color="gray.500" flexShrink={0}>{formatTime(record.recorded_at)}</Text></Flex>)}</Stack>}
    </>}
  </Box>;
}

function LifecycleStep({ number, icon, title, detail }) {
  return <Box className="metric-card"><Flex justify="space-between" align="center"><Flex boxSize="36px" bg="sage.100" color="sage.700" borderRadius="9px" align="center" justify="center"><Icon as={icon} /></Flex><Text className="eyebrow" fontSize="10px" color="gray.400">STEP {number}</Text></Flex><Text mt={5} fontWeight="700">{title}</Text><Text mt={2} fontSize="sm" color="gray.500" lineHeight="1.6">{detail}</Text></Box>;
}

function HookContract({ receives, does, impact }) {
  return <SimpleGrid columns={{ base: 1, md: 3 }} gap={3} mb={4}>
    <ContractItem label="RECEIVES" value={receives} />
    <ContractItem label="DOES" value={does} />
    <ContractItem label="CALLER IMPACT" value={impact} />
  </SimpleGrid>;
}

function ContractItem({ label, value }) {
  return <Box borderLeft="3px solid" borderColor="sage.400" pl={4} py={1}><Text className="eyebrow" fontSize="9px" color="sage.700">{label}</Text><Text mt={2} fontSize="sm" color="gray.600" lineHeight="1.6">{value}</Text></Box>;
}

function HookHeader({ number, title, description, badge, icon }) {
  return <Flex mt={12} mb={5} justify="space-between" align="end" gap={4}><HStack spacing={3}><Flex boxSize="36px" bg="sage.100" color="sage.700" borderRadius="9px" align="center" justify="center"><Icon as={icon} /></Flex><Box><Text className="eyebrow" fontSize="10px" color="sage.700">HOOK {number}</Text><Heading mt={1} fontSize="2xl" letterSpacing="-0.04em">{title}</Heading><Text mt={1} fontSize="sm" color="gray.500">{description}</Text></Box></HStack><Badge colorScheme="teal" borderRadius="full" px={3} py={1}>{badge}</Badge></Flex>;
}

function UsageStat({ label, value }) {
  return <Box bg="white" border="1px solid" borderColor="blackAlpha.200" borderRadius="12px" p={5}><Text className="eyebrow" fontSize="10px" color="gray.500">{label}</Text><Text mt={3} fontSize="xl" fontWeight="700">{value}</Text></Box>;
}

function UsageDatum({ label, value }) {
  return <Box><Text className="eyebrow" fontSize="9px" color="gray.500">{label}</Text><Text mt={1} fontSize="sm" fontWeight="700" whiteSpace="nowrap">{value}</Text></Box>;
}

function EmptyState({ icon, title, detail }) {
  return <Box mt={5} border="1px dashed" borderColor="sage.200" borderRadius="12px" p={{ base: 8, md: 12 }} textAlign="center" bg="whiteAlpha.600"><Icon as={icon} boxSize={7} color="sage.500" /><Text mt={4} fontWeight="700">{title}</Text><Text mt={2} fontSize="sm" color="gray.500">{detail}</Text></Box>;
}

function Evidence({ label, value }) {
  const rendered = typeof value === "string" ? value : JSON.stringify(value, null, 2);
  return <Box><Text className="eyebrow" fontSize="10px" color="gray.500" mb={2}>{label}</Text><Box bg="gray.50" borderRadius="9px" p={4} fontFamily="mono" fontSize="xs" whiteSpace="pre-wrap" maxH="240px" overflowY="auto">{rendered}</Box></Box>;
}