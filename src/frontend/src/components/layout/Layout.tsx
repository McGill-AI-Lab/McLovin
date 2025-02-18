import { Box } from "@mui/material";
import { Outlet, useLocation } from "react-router-dom";
import { Navbar } from "./Navbar";

interface LayoutProps {
  children?: React.ReactNode;
}

export const Layout = ({ children }: LayoutProps) => {
  const location = useLocation();
  const isHomePage = location.pathname === "/";

  return (
    <Box
      sx={{
        minHeight: "100vh",
        background: "linear-gradient(135deg, #ff9a9e, #fad0c4)",
      }}
    >
      {!isHomePage && <Navbar />}
      <Box component="main" sx={{ p: isHomePage ? 0 : 3 }}>
        {children || <Outlet />}
      </Box>
    </Box>
  );
};
