import { createTheme } from "@mui/material/styles";

export const theme = createTheme({
  palette: {
    primary: {
      main: "#C5050C", // McGill Red
      light: "#d4363c",
      dark: "#8c0309",
    },
    secondary: {
      main: "#2D2D2D",
      light: "#484848",
      dark: "#1A1A1A",
    },
    background: {
      default: "#FFFFFF",
      paper: "#FFFFFF",
    },
  },
  typography: {
    fontFamily: '"Inter", system-ui, -apple-system, sans-serif',
    h1: {
      fontWeight: 600,
    },
    h2: {
      fontWeight: 600,
    },
    button: {
      textTransform: "none",
      fontWeight: 500,
    },
  },
  shape: {
    borderRadius: 12,
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: "8px",
          padding: "10px 24px",
          boxShadow: "none",
        },
        contained: {
          "&:hover": {
            boxShadow: "none",
          },
        },
      },
    },
    MuiAppBar: {
      styleOverrides: {
        root: {
          backgroundColor: "#FFFFFF",
          boxShadow: "none",
          borderBottom: "1px solid #EAEAEA",
        },
      },
    },
  },
});
