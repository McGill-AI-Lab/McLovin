import {
  AppBar,
  Toolbar,
  Typography,
  IconButton,
  Box,
  Button,
  Avatar,
} from "@mui/material";
import { Link } from "react-router-dom";
import FavoriteIcon from "@mui/icons-material/Favorite";
import ChatIcon from "@mui/icons-material/Chat";
import PersonIcon from "@mui/icons-material/Person";

export const Navbar = () => {
  return (
    <AppBar position="fixed">
      <Toolbar
        sx={{
          padding: { xs: "0 16px", sm: "0 24px" }, // Reduce padding on mobile
          width: "100%",
          maxWidth: "100%",
          margin: "0",
        }}
      >
        <Typography
          variant="h6"
          component={Link}
          to="/"
          sx={{
            color: "primary.main",
            textDecoration: "none",
            fontSize: "24px",
            fontWeight: "700",
            letterSpacing: "-0.5px",
          }}
        >
          McLovin
        </Typography>

        <Box sx={{ flexGrow: 1 }} />

        <Box className="flex items-center gap-2 sm:gap-4">
          <IconButton
            component={Link}
            to="/matches"
            className="hover:bg-gray-50"
          >
            <FavoriteIcon sx={{ color: "secondary.main" }} />
          </IconButton>

          <IconButton
            component={Link}
            to="/messages"
            className="hover:bg-gray-50"
          >
            <ChatIcon sx={{ color: "secondary.main" }} />
          </IconButton>

          <IconButton
            component={Link}
            to="/profile"
            className="hover:bg-gray-50"
          >
            <Avatar sx={{ width: 32, height: 32 }}>
              <PersonIcon />
            </Avatar>
          </IconButton>
        </Box>
      </Toolbar>
    </AppBar>
  );
};
