import { Box, Typography, Button } from "@mui/material";
import { Link } from "react-router-dom";

export const Home = () => {
  return (
    <Box sx={{ textAlign: "center", py: 8 }}>
      <Typography variant="h2" component="h1" gutterBottom>
        Welcome to DateMcGill
      </Typography>
      <Typography variant="h5" component="h2" gutterBottom>
        Find your perfect match at McGill University
      </Typography>
      <Box sx={{ mt: 4 }}>
        <Button
          component={Link}
          to="/login"
          variant="contained"
          size="large"
          sx={{ mr: 2 }}
        >
          Login
        </Button>
        <Button component={Link} to="/demo" variant="outlined" size="large">
          View Demo
        </Button>
      </Box>
    </Box>
  );
};
