import { extendTheme } from "@chakra-ui/react";

export const theme = extendTheme({
  fonts: {
    heading: "'Space Grotesk', system-ui, sans-serif",
    body: "'Space Grotesk', system-ui, sans-serif",
  },
  colors: {
    ink: "#17131f",
    sage: { 50: "#ecfdfb", 100: "#ccfbf1", 500: "#0f9f92", 700: "#0f766e" },
    coral: { 50: "#fff0e8", 200: "#ffd0ba", 500: "#f15b2a" },
  },
  styles: {
    global: {
      "html, body": { background: "#f7f5f2", color: "#17131f" },
      body: { minWidth: "320px" },
    },
  },
  components: {
    Button: { baseStyle: { borderRadius: "10px", fontWeight: 700 } },
    Input: { defaultProps: { focusBorderColor: "sage.500" } },
    Textarea: { defaultProps: { focusBorderColor: "sage.500" } },
  },
});
